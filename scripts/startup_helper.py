import subprocess


def startup_helper():
    """
    Guides the user through the process of starting up the ADAM system.
    """
    print("Welcome to the ADAM System Startup Helper!")
    print("This script will help you to start up the ADAM system.")

    # Check if dependencies are installed
    try:
        import requests
        import yaml
    except ImportError:
        print("Dependencies are not installed. Please run 'pip install -r requirements.txt' to install them.")
        return

    # Start the system
    print("Starting the ADAM system...")
    subprocess.run(["python", "scripts/run_adam.py"])

if __name__ == "__main__":
    startup_helper()
