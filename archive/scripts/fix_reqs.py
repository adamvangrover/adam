import re

with open('requirements.txt', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Remove --index-url ...
    line = re.sub(r'\s*--index-url\s+\S+', '', line)
    new_lines.append(line)

with open('requirements_uv.txt', 'w') as f:
    f.writelines(new_lines)
