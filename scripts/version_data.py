import json
import argparse
from datetime import datetime

def version_data(file_path, version_type, change_description):
    """
    Increments the version number of a data file and adds an entry to the change log in VERSIONING.md.

    Args:
        file_path (str): The path to the data file.
        version_type (str): One of 'major', 'minor', or 'patch'.
        change_description (str): A description of the change.
    """
    with open('version_control.json', 'r+') as f:
        version_control = json.load(f)
        versions = version_control.get(file_path, '0.0.0').split('.')
        major, minor, patch = [int(v) for v in versions]

        if version_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif version_type == 'minor':
            minor += 1
            patch = 0
        elif version_type == 'patch':
            patch += 1

        new_version = f"{major}.{minor}.{patch}"
        version_control[file_path] = new_version
        f.seek(0)
        json.dump(version_control, f, indent=4)
        f.truncate()

    with open('VERSIONING.md', 'a') as f:
        today = datetime.now().strftime('%Y-%m-%d')
        f.write(f"\n*   **{new_version} ({today}):**\n")
        f.write(f"    *   {change_description}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Version a data file.')
    parser.add_argument('file_path', help='The path to the data file.')
    parser.add_argument('version_type', choices=['major', 'minor', 'patch'], help='The type of version increment.')
    parser.add_argument('change_description', help='A description of the change.')
    args = parser.parse_args()
    version_data(args.file_path, args.version_type, args.change_description)
