import json

def test_provenance():
    with open('showcase/data/adam_daily/2026-06-04/data.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            assert 'context_provenance' in data
            print(f"context_provenance is {data['context_provenance']}")

test_provenance()
