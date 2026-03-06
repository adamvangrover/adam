import os
import sys
import asyncio
import logging
import argparse

from core.symphony.orchestrator import SymphonyOrchestrator
from core.symphony.api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Symphony Orchestrator")
    parser.add_argument("workflow", nargs="?", default="WORKFLOW.md", help="Path to WORKFLOW.md")
    parser.add_argument("--port", type=int, help="Optional HTTP server port")
    args = parser.parse_args()

    workflow_path = os.path.abspath(args.workflow)
    if not os.path.exists(workflow_path):
        logger.error(f"Workflow file not found: {workflow_path}")
        sys.exit(1)

    orchestrator = SymphonyOrchestrator(workflow_path)
    if not orchestrator.reload_config():
        logger.error("Failed startup validation, check configuration.")
        sys.exit(1)

    # Determine port
    port = args.port
    if port is None and orchestrator.config:
        port = orchestrator.config.server_port

    # Start orchestrator logic
    logger.info(f"Starting Symphony Orchestrator using {workflow_path}")
    await orchestrator.start()

    if port is not None:
        logger.info(f"Starting HTTP API on port {port}")
        app = create_app(orchestrator)

        # In a real async environment we would use an ASGI server like uvicorn or hypercorn.
        # Here we just use Werkzeug for demonstration/simplicity, which blocks.
        # To run alongside async orchestrator, we would run it in a thread.
        import threading
        from werkzeug.serving import make_server

        # We need a small wrapper to allow shutting it down
        class ServerThread(threading.Thread):
            def __init__(self, app_to_run, p):
                threading.Thread.__init__(self)
                self.server = make_server('127.0.0.1', p, app_to_run)
                self.ctx = app_to_run.app_context()
                self.ctx.push()

            def run(self):
                logger.info(f"Server started on 127.0.0.1:{self.server.port}")
                self.server.serve_forever()

            def shutdown(self):
                self.server.shutdown()

        server_thread = ServerThread(app, port)
        server_thread.daemon = True
        server_thread.start()

    try:
        # Keep main thread alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        orchestrator._running = False
        sys.exit(0)

def cli_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    cli_entry()
