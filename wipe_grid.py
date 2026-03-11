import os

filepath = "showcase/market_mayhem_archive.html"
with open(filepath, "r") as f:
    content = f.read()

start_tag = '<div id="archiveGrid" style="padding-bottom: 50px;">'
end_tag = '        </main>'

start_idx = content.find(start_tag)
end_idx = content.find(end_tag)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx + len(start_tag)] + """
                <div style="text-align:center; padding: 50px; color: #666;">
                    <i class="fas fa-spinner fa-spin fa-2x"></i><br>
                    Loading Archive Index...
                </div>
            </div>

""" + content[end_idx:]
    with open(filepath, "w") as f:
        f.write(new_content)
    print("Successfully wiped archiveGrid.")
else:
    print("Failed to find boundaries.")
