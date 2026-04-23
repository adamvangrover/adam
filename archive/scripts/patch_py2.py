import re

with open("scripts/generate_real_unified_memos.py", "r") as f:
    py = f.read()

# Add projected_rev
py = py.replace("""        base_fcf = fcf / 1e6
        base_rev = projected_rev[0]""", """        base_fcf = fcf / 1e6
        rev_last = hist_records[0]["revenue"] if hist_records else 1000
        projected_rev = [rev_last * (1 + growth), rev_last * (1 + growth)**2]
        base_rev = projected_rev[0]""")

with open("scripts/generate_real_unified_memos.py", "w") as f:
    f.write(py)
