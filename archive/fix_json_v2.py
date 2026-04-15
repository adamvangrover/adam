
import re

filepath = 'core/libraries_and_archives/reports/software_industry_report_20250225.json'

with open(filepath, 'r') as f:
    content = f.read()

section_marker = '"title": "Software Industry Trends: Adapting to the AI-Powered Cloud",'
start_idx = content.find(section_marker)

if start_idx != -1:
    content_key = '"content": "'
    c_start = content.find(content_key, start_idx)
    if c_start != -1:
        c_start += len(content_key)

        next_section = '"title": "Sub-Industry Analysis: Transformation Across the Board"'
        next_idx = content.find(next_section, c_start)

        if next_idx != -1:
            obj_end = content.rfind('    },', c_start, next_idx)
            if obj_end != -1:
                # Capture chunk
                chunk = content[c_start:obj_end]

                # Check for signature
                if "SaaS Metrics and Accounting" in chunk:
                    # Fix newlines
                    fixed_chunk = chunk.replace('\n', '\\n')

                    # Ensure it ends with quote.
                    # Original `chunk` ends with `\n\n`.
                    # `fixed_chunk` ends with `\\n\\n`.

                    # We need to construct: `...\\n\\n` + `"\n` + `    },`

                    new_content = content[:c_start] + fixed_chunk + '"\n' + content[obj_end:]

                    with open(filepath, 'w') as f:
                        f.write(new_content)
                    print("Fixed.")
                else:
                    print("Chunk does not match expected content.")
            else:
                print("Could not find object end.")
        else:
            print("Could not find next section.")
    else:
        print("Could not find content start.")
else:
    print("Could not find section start.")
