
import os
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from services.webapp.config import config
from services.webapp.api import create_app

def check_debug_mode():
    print("Checking default configuration...")
    # Simulate no env var
    if 'FLASK_CONFIG' in os.environ:
        del os.environ['FLASK_CONFIG']

    app_default = create_app('default')
    print(f"Default Config DEBUG: {app_default.config['DEBUG']}")

    if app_default.config['DEBUG']:
        print("FAIL: Default config has DEBUG=True")
    else:
        print("PASS: Default config has DEBUG=False")

if __name__ == "__main__":
    check_debug_mode()
