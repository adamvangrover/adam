import json
import os
import re
from bs4 import BeautifulSoup

def evaluate_factuality(extracted_data):
    """
    Evaluates the factuality of the extracted data points using heuristics.
    Returns a dictionary containing the score, is_factual boolean, and a list of reasons.
    """
    score = 100
    reasons = []

    # Check for placeholder terms
    for key, value in extracted_data.items():
        if value and isinstance(value, str):
            if re.search(r'\b(unknown|tbd|mock|test|dummy)\b', value, re.IGNORECASE):
                score -= 20
                reasons.append(f"Found placeholder term in {key}")

    # Numeric constraints
    if extracted_data.get('conviction') is not None:
        try:
            val = float(str(extracted_data['conviction']).replace('%', '').strip())
            if not (0 <= val <= 100):
                score -= 20
                reasons.append("Conviction score out of bounds (must be 0-100)")
        except ValueError:
            score -= 10
            reasons.append("Invalid conviction format")

    if extracted_data.get('sentiment') is not None:
        try:
            val = float(str(extracted_data['sentiment']).replace('%', '').strip())
            if not (0 <= val <= 100):
                score -= 20
                reasons.append("Sentiment score out of bounds (must be 0-100)")
        except ValueError:
            score -= 10
            reasons.append("Invalid sentiment format")
            
    if extracted_data.get('price') is not None:
        try:
            val = float(str(extracted_data['price']).replace(',', '').replace('$', '').strip())
            if val < 0:
                score -= 20
                reasons.append("Price cannot be negative")
        except ValueError:
            score -= 10
            reasons.append("Invalid price format")

    # Overall evaluation
    is_factual = score >= 80
    return {
        "score": score,
        "is_factual": is_factual,
        "reasons": reasons
    }

def extract_from_text(text):
    """
    Extracts conviction, sentiment, price, and outlook from plain text (or markdown).
    """
    # Look for "Conviction" followed by an optional colon/space, then a number, optional "/100" and optional "%"
    conviction_match = re.search(r'Conviction(?:[\s:]*)?(\d{1,3}(?:\.\d+)?)(?:/?(?:100)?)?%?', text, re.IGNORECASE)
    # Same for Sentiment
    sentiment_match = re.search(r'Sentiment(?:[\s:]*)?(\d{1,3}(?:\.\d+)?)(?:/?(?:100)?)?%?', text, re.IGNORECASE)
    # Look for "Price", "Target", or "Value" followed by an optional "$" and a number with commas/decimals
    price_match = re.search(r'(?:Price|Target|Value)(?:[\s:]*)?\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    # Extract outlook up to the next newline or period
    outlook_match = re.search(r'Outlook(?:[\s:]*)(.+?)(?:\n|\.|$)', text, re.IGNORECASE)
    
    return {
        "conviction": conviction_match.group(1) if conviction_match else None,
        "sentiment": sentiment_match.group(1) if sentiment_match else None,
        "price": price_match.group(1) if price_match else None,
        "outlook": outlook_match.group(1).strip() if outlook_match else None
    }

def extract_from_html(html_content):
    """
    Parses HTML to extract the text, then uses text extraction heuristics.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    return extract_from_text(text)

def process_file(filepath):
    """
    Reads the file, extracts data, evaluates it, and appends a summary block if not already present.
    Returns a dictionary of the result.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    if filepath.endswith('.html'):
        extracted = extract_from_html(content)
    elif filepath.endswith('.md'):
        extracted = extract_from_text(content)
    else:
        return None

    evaluation = evaluate_factuality(extracted)
    
    # Check if already evaluated to avoid duplicates
    if "### Data Accuracy Evaluation" in content or "<!-- Data Accuracy Evaluation -->" in content:
        print(f"Skipping {filepath} (already evaluated)")
        return None

    if filepath.endswith('.html'):
        # For HTML, append a structured div block
        html_summary = f"""
<!-- Data Accuracy Evaluation -->
<div style="margin-top: 2rem; padding: 1rem; border-top: 2px solid #333; font-family: monospace;">
    <h3>Data Accuracy Evaluation</h3>
    <ul>
        <li><strong>Score:</strong> {evaluation['score']}/100</li>
        <li><strong>Is Factual:</strong> {evaluation['is_factual']}</li>
        <li><strong>Extracted Data:</strong> {json.dumps(extracted).replace('<', '&lt;').replace('>', '&gt;')}</li>
        <li><strong>Reasons:</strong> {', '.join(evaluation['reasons']) if evaluation['reasons'] else 'None'}</li>
    </ul>
</div>
"""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(html_summary)
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")
            return None
    else:
        # For Markdown, append a structured text block
        summary_block = f"""

---
### Data Accuracy Evaluation
*   **Score:** {evaluation['score']}/100
*   **Is Factual:** {evaluation['is_factual']}
*   **Extracted Data:** {json.dumps(extracted)}
*   **Reasons:** {', '.join(evaluation['reasons']) if evaluation['reasons'] else 'None'}
"""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(summary_block)
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")
            return None

    return {
        "filepath": filepath,
        "extracted_data": extracted,
        "evaluation": evaluation
    }

def main():
    directories = ['showcase', 'core/libraries_and_archives', 'exports/market_mayhem']
    files_to_process = []
    
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.html') or file.endswith('.md'):
                    files_to_process.append(os.path.join(root, file))
    
    report_data = []
    print(f"Found {len(files_to_process)} files to evaluate.")
    
    for file in files_to_process:
        result = process_file(file)
        if result:
            report_data.append(result)
            
    report_path = 'showcase/data/accuracy_evaluation_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        print(f"Evaluated {len(report_data)} files. Report saved to {report_path}.")
    except Exception as e:
        print(f"Error saving report to {report_path}: {e}")

if __name__ == '__main__':
    main()
