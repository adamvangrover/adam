// Nexus Template Library
// Reusable text templates for the Notepad application
window.TEMPLATE_LIBRARY = {
    "meeting_notes": {
        "name": "Meeting Notes",
        "content": "MEETING NOTES\nDate: [Date]\nAttendees: [Names]\n\nAGENDA:\n1. \n2. \n3. \n\nACTION ITEMS:\n- [ ] \n- [ ] \n"
    },
    "incident_report": {
        "name": "Incident Report",
        "content": "INCIDENT REPORT\nID: #INC-[ID]\nSeverity: [Low/Medium/High]\n\nDESCRIPTION:\n[Describe what happened]\n\nROOT CAUSE:\n[Analysis]\n\nMITIGATION:\n[Steps taken]"
    },
    "code_snippet": {
        "name": "Python Snippet",
        "content": "def analyze_market(data):\n    \"\"\"\n    Analyzes market data for anomalies.\n    \"\"\"\n    results = []\n    for item in data:\n        if item['risk_score'] > 80:\n            results.append(item)\n    return results\n"
    },
    "memo": {
        "name": "Internal Memo",
        "content": "MEMORANDUM\nTo: All Staff\nFrom: [Sender]\nDate: [Date]\nSubject: [Subject]\n\n[Body of memo]\n\nRegards,\n[Sender Name]"
    }
};
