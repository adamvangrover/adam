from bs4 import BeautifulSoup

def find_tag_lines(filename):
    with open(filename, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    def find_closing_div(start_line_idx):
        stack = 0
        for i in range(start_line_idx, len(lines)):
            if '<div' in lines[i]:
                stack += lines[i].count('<div')
            if '</div' in lines[i]:
                stack -= lines[i].count('</div')
                if stack == 0:
                    return i + 1
        return -1

    for i, line in enumerate(lines):
        if 'id="desktop"' in line:
            print(f'id="desktop" at line {i+1}, closes at {find_closing_div(i)}')
        if 'id="start-menu"' in line:
            print(f'id="start-menu" at line {i+1}, closes at {find_closing_div(i)}')
        if 'fixed top-0' in line:
            print(f'fixed top-0 at line {i+1}, closes at {find_closing_div(i)}')

print("--- sovereign_os.html ---")
find_tag_lines("showcase/sovereign_os.html")
