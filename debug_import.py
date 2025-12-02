import sys
import os
sys.path.append(os.getcwd())
try:
    import core
    print(f"Core path: {core.__path__}")
    import core.data_processing
    print(f"Data Processing path: {core.data_processing.__path__}")
    print(f"Directory listing of data processing: {os.listdir(core.data_processing.__path__[0])}")

    from core.data_processing import gold_standard_scrubber
    print("Imported gold_standard_scrubber successfully")
except ImportError as e:
    print(f"Failed to import: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
