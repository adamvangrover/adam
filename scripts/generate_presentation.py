import sys
import os
import re

def generate_html_from_markdown(md_filepath, html_filepath):
    if not os.path.exists(md_filepath):
        print(f"Error: Could not find {md_filepath}")
        sys.exit(1)

    with open(md_filepath, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Pre-process the markdown to insert --- as a slide separator
    # This prevents the slide titles from being eaten by reveal.js's data-separator regex
    processed_md = re.sub(r'^(Slide [0-9]+:.*)$', r'---\n\1', md_content, flags=re.MULTILINE)

    html_content = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Next-Generation Agentic Automation for Credit Risk Control</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reset.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/theme/black.min.css">
  </head>
  <body>
    <div class="reveal">
      <div class="slides">
        <section data-markdown
                 data-separator="^---$"
                 data-separator-notes="^Speaker Notes \\(CRO\\):">
          <textarea data-template>
{processed_md}
          </textarea>
        </section>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/reveal.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/plugin/markdown/markdown.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.3.1/plugin/notes/notes.min.js"></script>
    <script>
      Reveal.initialize({{
        hash: true,
        plugins: [ RevealMarkdown, RevealNotes ]
      }});
    </script>
  </body>
</html>
"""

    with open(html_filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Successfully generated {html_filepath}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        md_filepath = sys.argv[1]
        html_filepath = sys.argv[2]
    else:
        md_filepath = "docs/presentations/Agentic_Automation_Credit_Risk.md"
        html_filepath = "docs/presentations/Agentic_Automation_Credit_Risk.html"

    generate_html_from_markdown(md_filepath, html_filepath)
