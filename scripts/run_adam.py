import sys
import os

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.main import main

if __name__ == "__main__":
    main()
