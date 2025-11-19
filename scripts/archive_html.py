import os
import json

def archive_html_files():
    """
    Scans the entire repository for all .html files, copies them to a new
    directory named docs/ui_archive_v1/, flattens the folder structure,
    renames them clearly, and generates a manifest.json file.
    """
    archive_dir = "docs/ui_archive_v1"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    manifest = {}
    for root, dirs, files in os.walk("."):
        if archive_dir in root:
            continue
        for file in files:
            if file.endswith(".html"):
                original_path = os.path.join(root, file)
                new_filename = f"{os.path.basename(root)}_{file}".replace(
                    "/", "_"
                ).replace("\\", "_")
                new_filepath = os.path.join(archive_dir, new_filename)
                
                # Copy the file
                with open(original_path, "r", encoding="utf-8") as f_in, open(
                    new_filepath, "w", encoding="utf-8"
                ) as f_out:
                    f_out.write(f_in.read())

                manifest[new_filename] = original_path
    
    with open(
        os.path.join(archive_dir, "manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=4)

if __name__ == "__main__":
    archive_html_files()
