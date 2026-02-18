import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

print(f"[*] Starting market_mayhem module on port {PORT}...")
print(f"[*] Opening browser to http://localhost:{PORT}")

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()
except OSError:
    print(f"[!] Port {PORT} in use. Try running: python3 -m http.server")