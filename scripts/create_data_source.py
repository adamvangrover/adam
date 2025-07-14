import os
import sys

def create_data_source(data_source_name):
    """
    Creates a new data source file in the core/data_sources directory.

    Args:
        data_source_name (str): The name of the data source to create.
    """
    data_source_file_path = f"core/data_sources/{data_source_name.lower()}_source.py"
    if os.path.exists(data_source_file_path):
        print(f"Data source file already exists: {data_source_file_path}")
        return

    with open(data_source_file_path, "w") as f:
        f.write(f"""from core.data_access.base_data_source import BaseDataSource

class {data_source_name}Source(BaseDataSource):
    def __init__(self):
        super().__init__("{data_source_name}Source")

    def get_data(self, query):
        # Implement data retrieval logic here
        pass
""")

    print(f"Data source file created successfully: {data_source_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_data_source.py <data_source_name>")
        sys.exit(1)
    data_source_name = sys.argv[1]
    create_data_source(data_source_name)
