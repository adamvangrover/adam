import json
import random
import sys
import os
import re

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion.semantic_chunker import SemanticChunker
from core.evaluation.llm_judge import ConvictionScorer

def load_newsletter_data(filepath):
    """Loads newsletter data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Skipping newsletter data.")
        return []

def clean_html(text):
    """Removes HTML tags from text."""
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

def generate_stream():
    chunker = SemanticChunker()
    judge = ConvictionScorer()

    # Base scenarios (Hardcoded)
    scenarios = [
        {
            "topic": "Market Mayhem: 2026 Outlook",
            "text": "The 2026 market landscape is defined by the 'Reflationary Agentic Boom'. Inflation is sticky at 2.3% due to tariffs. However, real GDP is growing at 2.6%. The 'Sovereign AI' capex cycle is providing a floor for hardware demand.",
            "questions": ["What is driving the 2026 economic boom?", "Why is inflation sticky?"],
            "answers": ["The boom is driven by fiscal stimulus and 'Sovereign AI' capex.", "Inflation is sticky due to geopolitical fragmentation and tariffs."]
        },
        {
            "topic": "Nvidia & The Jevons Paradox",
            "text": "Nvidia's stock stabilized as the Jevons Paradox took hold. Cheaper inference costs from models like DeepSeek R1 led to a massive increase in agentic workloads. This increased total compute demand despite per-unit efficiency gains.",
            "questions": ["Explain the Jevons Paradox in this context."],
            "answers": ["Lower inference costs increased total demand for compute by enabling more agentic workloads."]
        },
        {
            "topic": "Crypto Institutionalization",
            "text": "FASB ASU 2023-08 changed the game for corporate treasuries. Bitcoin is now measured at fair value. Rumors suggest Qatar (QIA) is acquiring BTC to diversify reserves. Peru might be mining using hydro energy.",
            "questions": ["What did FASB ASU 2023-08 change?", "Who is rumored to be buying BTC?"],
            "answers": ["It allowed companies to report crypto at fair value, boosting net income during rallies.", "Qatar (QIA) is rumored to be buying."]
        }
    ]

    # Ingest Newsletter Data
    newsletter_path = "showcase/data/newsletter_data.json"
    newsletters = load_newsletter_data(newsletter_path)

    for item in newsletters:
        full_text = clean_html(item.get("full_body", ""))
        summary = item.get("summary", "")
        title = item.get("title", "Market Update")
        date = item.get("date", "Unknown Date")

        # Create synthetic Q&A templates
        questions = []
        answers = []

        # Template 1: Summary
        if summary:
            questions.append(f"Summarize the key market events on {date}.")
            answers.append(summary)

            questions.append(f"What is the main takeaway from '{title}'?")
            answers.append(summary)

        # Template 2: Title significance
        if title:
             questions.append(f"What happened during '{title}'?")
             answers.append(summary if summary else f"Events related to {title} on {date}.")

        if full_text and questions:
            scenarios.append({
                "topic": f"{title} ({date})",
                "text": full_text,
                "questions": questions,
                "answers": answers
            })

    output_path = "showcase/data/artisanal/teacher_student_stream.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for i, sc in enumerate(scenarios):
            # 1. Chunking
            chunks = chunker.split_text(sc['text'])

            for chunk in chunks:
                if not chunk.strip():
                    continue

                # 2. Scoring (LLM Judge)
                conviction = judge.score(chunk)

                # 3. Generate Teacher-Student Pairs
                # Ensure Q and A match by index to maintain semantic consistency
                if not sc['questions']:
                    continue

                idx = random.randint(0, len(sc['questions']) - 1)
                q = sc['questions'][idx]
                a = sc['answers'][idx]

                entry = {
                    "id": f"sim_{i}_{random.randint(1000,9999)}",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ],
                    "context_chunk": chunk,
                    "metadata": {
                        "topic": sc['topic'],
                        "conviction_score": conviction['score'],
                        "rationale": conviction['rationale'],
                        "judge": conviction['judge_model'],
                        "timestamp": "2026-02-18T12:00:00Z" # Using fixed timestamp for simulation coherence
                    }
                }

                f.write(json.dumps(entry) + '\n')

    print(f"Generated {len(scenarios)} scenarios (Base + Newsletters) into {output_path}")

if __name__ == "__main__":
    generate_stream()
