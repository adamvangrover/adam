import re

with open("showcase/js/comprehensive_credit_dashboard.js", "r") as f:
    js = f.read()

# Update renderSidebar to support memo.companyName
js = re.sub(
    r"const ticker = memo\._metadata\?\.ticker \|\| memo\.borrower_name;",
    r"const ticker = memo._metadata?.ticker || memo.ticker || memo.borrower_name || 'N/A';",
    js
)
js = re.sub(
    r"const name = memo\.borrower_name;",
    r"const name = memo.borrower_name || memo.companyName || 'Unknown';",
    js
)
# Update search input
js = re.sub(
    r"const name = \(m\.borrower_name \|\| ''\)\.toLowerCase\(\);",
    r"const name = (m.borrower_name || m.companyName || '').toLowerCase();",
    js
)
js = re.sub(
    r"const ticker = \(m\._metadata\?\.ticker \|\| ''\)\.toLowerCase\(\);",
    r"const ticker = (m._metadata?.ticker || m.ticker || '').toLowerCase();",
    js
)


# Update selectMemo mapping
js = js.replace("document.getElementById('mc-name').textContent = memo.borrower_name || 'Unknown';",
                "document.getElementById('mc-name').textContent = memo.borrower_name || memo.companyName || 'Unknown';")
js = js.replace("document.getElementById('mc-ticker').textContent = memo._metadata?.ticker || 'N/A';",
                "document.getElementById('mc-ticker').textContent = memo._metadata?.ticker || memo.ticker || 'N/A';")
js = js.replace("document.getElementById('mc-sector').textContent = memo._metadata?.sector || memo.borrower_details?.sector || 'Unknown';",
                "document.getElementById('mc-sector').textContent = memo._metadata?.sector || memo.sector || memo.borrower_details?.sector || 'Unknown';")

# Risk score fallback
js = js.replace("document.getElementById('mc-risk-score').textContent = memo._metadata?.risk_score || memo.risk_score || '--';",
                "document.getElementById('mc-risk-score').textContent = memo._metadata?.risk_score || memo.risk_score || (memo.financials?.historicals?.net_debt_to_ebitda < 2 ? 85 : 60) || '--';")

# Exec summary
js = js.replace("document.getElementById('mc-summary').innerHTML = (memo.executive_summary || 'No summary available.').replace(/\\n/g, '<br>');",
                "document.getElementById('mc-summary').innerHTML = (memo.executive_summary || memo.companyName + ' operates in ' + (memo.sector || 'the market') + ' sector.').replace(/\\n/g, '<br>');")


with open("showcase/js/comprehensive_credit_dashboard.js", "w") as f:
    f.write(js)
