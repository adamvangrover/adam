# Chain of Verification: Fact Check & Revision

**Version:** 1.0
**Role:** Editor-in-Chief / Fact Checker
**Task:** Verify the accuracy of a generated financial report.

---

## 1. Input Text
"{{draft_text}}"

## 2. Verification Protocol
Perform the following "Chain of Verification":

1.  **Extraction:** List all quantitative claims (numbers, dates, percentages) and qualitative assertions (names, roles, events) in the text.
2.  **Verification:** For each claim, check against your internal knowledge base or the provided {{source_documents}}.
    *   *If true:* Mark as [VERIFIED].
    *   *If false or uncertain:* Mark as [FLAGGED].
3.  **Revision:** Rewrite the text.
    *   Correct all [FLAGGED] items.
    *   Remove any claims that cannot be verified.
    *   Maintain the original tone and structure.

## 3. Output Format
Return a JSON object:

```json
{
  "original_score": <0-100 accuracy score>,
  "flags": [
    {"claim": "Revenue was $5B", "verdict": "False", "correction": "$4.2B"}
  ],
  "revised_text": "..."
}
```
