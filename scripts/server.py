import http.server
import socketserver
import json
import sys
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGServer")

# Ensure we can import from scripts
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from run_credit_memo_rag import CreditMemoRAGPipeline
except ImportError as e:
    logger.error(f"Failed to import CreditMemoRAGPipeline: {e}")
    # Define a dummy class to prevent crash on start, but will fail on execution
    class CreditMemoRAGPipeline:
        def __init__(self, ticker): pass
        async def run(self, document_path):
            raise ImportError("RAG Pipeline dependencies missing")

PORT = 8000

class RAGRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/run_rag':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                ticker = data.get('ticker')
                file_path = data.get('file')

                if not ticker or not file_path:
                    self.send_error(400, "Missing ticker or file path")
                    return

                # SECURITY: Path Traversal Prevention
                # Restrict file access to the 'data' directory
                allowed_dir = os.path.abspath("data")
                requested_path = os.path.abspath(file_path)

                # Ensure allowed_dir ends with separator to prevent partial matching (e.g. /data matching /database)
                if not allowed_dir.endswith(os.sep):
                    allowed_dir += os.sep

                if not requested_path.startswith(allowed_dir):
                    logger.warning(f"Security Alert: Path traversal attempt blocked. Input: {file_path}")
                    self.send_error(403, "Access denied: Invalid file path")
                    return

                if not os.path.exists(requested_path):
                    self.send_error(404, "File not found")
                    return

                logger.info(f"Received RAG request for {ticker} using {file_path}")

                # Execute Pipeline
                pipeline = CreditMemoRAGPipeline(ticker)

                # Run async pipeline in sync context
                try:
                    spread, memo = asyncio.run(pipeline.run(file_path))
                except RuntimeError:
                    # Handle case where event loop is already running (unlikely here but good practice)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    spread, memo = loop.run_until_complete(pipeline.run(file_path))

                if spread and memo:
                    response = {
                        "status": "success",
                        "message": f"RAG analysis complete for {ticker}",
                        "spread": spread,
                        "memo": memo
                    }
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                else:
                    self.send_error(500, "Pipeline failed to generate data")

            except Exception as e:
                logger.error(f"Error processing RAG request: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        # Serve static files from root
        # If path starts with /showcase/, serve normally.
        # If it is root /, serve showcase/index.html or similar if desired, but
        # default behavior is fine for now as we want to access specific pages.
        return super().do_GET()

# Set the working directory to the repo root for file serving
# This ensures that http://localhost:8000/showcase/sovereign_dashboard.html works
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with socketserver.TCPServer(("", PORT), RAGRequestHandler) as httpd:
    print(f"Serving at port {PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
