import re
import sys

def check_vulnerability():
    with open('showcase/js/nav.js', 'r') as f:
        content = f.read()

    # Pattern for _renderSearchResults XSS
    search_pattern = r"container\.innerHTML = results\.map\(r => `.*?\$\{r\.title\}.*?`\)\.join\(''\);"

    # Pattern for _renderSafeMode XSS
    safe_mode_pattern = r"safeNav\.innerHTML = `.*?NAV SYSTEM FAILURE: \$\{error\.message\}`;"

    vulnerable = False

    if re.search(search_pattern, content, re.DOTALL):
        print("VULNERABLE: _renderSearchResults uses innerHTML with r.title")
        vulnerable = True
    else:
        print("SAFE: _renderSearchResults pattern not found")

    if re.search(safe_mode_pattern, content, re.DOTALL):
        print("VULNERABLE: _renderSafeMode uses innerHTML with error.message")
        vulnerable = True
    else:
        print("SAFE: _renderSafeMode pattern not found")

    if vulnerable:
        print("Vulnerability confirmed.")
        sys.exit(0)
    else:
        print("No vulnerabilities found.")
        sys.exit(1)

if __name__ == "__main__":
    check_vulnerability()
