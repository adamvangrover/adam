import os
import json
import base64
import gzip

def get_file_content(filepath, ext):
    try:
        # text files
        text_exts = {
            '.html', '.htm', '.js', '.jsx', '.ts', '.tsx', '.json', '.jsonl',
            '.py', '.md', '.txt', '.yaml', '.yml', '.css', '.scss', '.sass',
            '.ttl', '.csv', '.go', '.sh', '.bash', '.zsh', '.toml', '.sql',
            '.j2', '.rs', '.xml', '.svg', '.ini', '.cfg', '.conf'
        }
        if ext.lower() in text_exts or ext == 'no_ext':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read(), "text"
        else:
            with open(filepath, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8'), "base64"
    except UnicodeDecodeError:
        try:
             with open(filepath, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8'), "base64"
        except:
             return None, "error"
    except Exception:
        return None, "error"

def build_tree(root_dir):
    tree = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.git' in dirpath or '.venv' in dirpath or '__pycache__' in dirpath:
            continue

        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == '.':
            parts = []
        else:
            parts = rel_path.split(os.sep)

        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]

        for f in filenames:
            if f in ['pack_repo.py', 'adam_repo_standalone.html']:
                continue
            current[f] = os.path.join(dirpath, f)

    return tree

def generate_html(output_file="adam_repo_standalone.html"):
    repo_root = os.getcwd()
    tree = build_tree(repo_root)

    files_data = {}

    for root, dirs, files in os.walk(repo_root):
        if '.git' in root or '.venv' in root or '__pycache__' in root:
            continue

        for file in files:
            if file in ['pack_repo.py', 'adam_repo_standalone.html']:
                continue
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, repo_root)
            ext = os.path.splitext(file)[1] or 'no_ext'

            # Skip massive files (over 5MB)
            if os.path.getsize(filepath) > 5 * 1024 * 1024:
                files_data[rel_path] = {"type": "toolarge"}
                continue

            content, ctype = get_file_content(filepath, ext)

            if content is None:
                files_data[rel_path] = {"type": "error"}
            else:
                files_data[rel_path] = {"type": ctype, "content": content}

    print(f"Collected {len(files_data)} files")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>Adam Repo Standalone</title>\n")
        f.write("""<style>
  body { display: flex; height: 100vh; margin: 0; font-family: sans-serif; }
  #sidebar { width: 300px; border-right: 1px solid #ccc; overflow-y: auto; padding: 10px; background: #f9f9f9; resize: horizontal; }
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #header { padding: 10px; border-bottom: 1px solid #ccc; background: #eee; font-weight: bold; }
  #content { flex: 1; overflow: auto; padding: 10px; white-space: pre-wrap; font-family: monospace; }
  .dir { font-weight: bold; cursor: pointer; margin-top: 5px; }
  .file { cursor: pointer; margin-left: 15px; color: #0066cc; font-size: 0.9em; }
  .file:hover { text-decoration: underline; background-color: #eef; }
  .hidden { display: none; }
  .dir-contents { margin-left: 10px; border-left: 1px dashed #ccc; padding-left: 5px; }
  #loading { padding: 20px; font-weight: bold; color: #666; }
</style>
""")
        f.write("</head>\n<body>\n")
        f.write("<div id=\"sidebar\">\n")
        f.write("  <h3>Adam Repository</h3>\n")
        f.write("  <div><button onclick=\"expandAll()\">Expand All</button> <button onclick=\"collapseAll()\">Collapse All</button></div>\n")
        f.write("  <hr>\n")

        def render_tree(t, prefix=""):
            res = ""
            for k, v in sorted(t.items()):
                if isinstance(v, dict):
                    res += f'<div class="dir" onclick="toggleDir(this)">📁 {k}</div>\n'
                    res += f'<div class="dir-contents hidden">\n'
                    res += render_tree(v, prefix + k + "/")
                    res += '</div>\n'
                else:
                    rel_path = os.path.relpath(v, repo_root)
                    js_path = rel_path.replace("'", "\\'").replace('"', '&quot;')
                    res += f"<div class=\"file\" onclick=\"showFile('{js_path}')\">📄 {k}</div>\n"
            return res

        f.write(render_tree(tree))

        f.write("</div>\n")
        f.write("<div id=\"main\">\n")
        f.write("  <div id=\"header\">Select a file to view</div>\n")
        f.write("  <div id=\"content\"><div id=\"loading\">Loading file data...</div></div>\n")
        f.write("</div>\n")

        json_data = json.dumps(files_data).encode('utf-8')
        compressed_data = gzip.compress(json_data)
        b64_data = base64.b64encode(compressed_data).decode('utf-8')

        f.write(f"<script id=\"repo-data\" type=\"text/plain\">\n{b64_data}\n</script>\n")

        f.write("""<script>
  let fileData = {};

  async function loadData() {
      try {
          const b64 = document.getElementById('repo-data').textContent.trim();
          const binaryString = window.atob(b64);
          const len = binaryString.length;
          const bytes = new Uint8Array(len);
          for (let i = 0; i < len; i++) {
              bytes[i] = binaryString.charCodeAt(i);
          }

          if ('DecompressionStream' in window) {
              const ds = new DecompressionStream('gzip');
              const stream = new Response(bytes).body.pipeThrough(ds);
              const response = new Response(stream);
              const blob = await response.blob();
              const text = await blob.text();
              fileData = JSON.parse(text);
          } else {
              document.getElementById('loading').innerText = "Browser doesn't support native DecompressionStream. Cannot load data.";
              return;
          }

          document.getElementById('loading').style.display = 'none';
      } catch (e) {
          document.getElementById('loading').innerText = "Error loading data: " + e;
          console.error(e);
      }
  }

  loadData();

  function expandAll() {
      document.querySelectorAll('.dir-contents').forEach(el => el.classList.remove('hidden'));
  }
  function collapseAll() {
      document.querySelectorAll('.dir-contents').forEach(el => el.classList.add('hidden'));
  }
  function toggleDir(el) {
      const contents = el.nextElementSibling;
      if (contents && contents.classList.contains('dir-contents')) {
          contents.classList.toggle('hidden');
      }
  }

  function showFile(path) {
      document.getElementById('header').innerText = path;
      const contentEl = document.getElementById('content');

      if (Object.keys(fileData).length === 0) {
          contentEl.innerText = "Data is still loading or failed to load...";
          return;
      }

      const data = fileData[path];
      if (!data) {
          contentEl.innerText = "Error: File data not found";
      } else if (data.type === "text") {
          contentEl.innerText = data.content;
      } else if (data.type === "base64") {
          if (path.match(/\\.(png|jpg|jpeg|gif|svg)$/i)) {
             contentEl.innerHTML = '<img src="data:image/' + path.split('.').pop() + ';base64,' + data.content + '" style="max-width: 100%;">';
          } else {
             contentEl.innerText = "[Binary File: " + path + "]\\nBase64 content hidden for brevity.\\nSize: " + data.content.length + " bytes";
          }
      } else if (data.type === "toolarge") {
          contentEl.innerText = "[Large File: " + path + "]\\nFile is over 5MB and was excluded.";
      } else {
          contentEl.innerText = "[Binary/Error File: " + path + "]\\nContent not available.";
      }
  }
</script>
</body>
</html>
""")

    print(f"Done creating {output_file}")

if __name__ == "__main__":
    generate_html()
