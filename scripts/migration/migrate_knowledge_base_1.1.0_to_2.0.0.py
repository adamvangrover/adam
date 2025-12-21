import json


def migrate_knowledge_base():
    """
    Migrates the knowledge_base.json file from version 1.1.0 to 2.0.0.
    """
    with open('data/knowledge_base.json', 'r+') as f:
        knowledge_base = json.load(f)

        # Add a new "Industry" section
        knowledge_base['Industry'] = {
            "description": "Information about different industries.",
            "Technology": {"description": "The technology industry."},
            "Finance": {"description": "The finance industry."}
        }

        # Rename "Valuation" to "ValuationMethods"
        if 'Valuation' in knowledge_base:
            knowledge_base['ValuationMethods'] = knowledge_base.pop('Valuation')

        f.seek(0)
        json.dump(knowledge_base, f, indent=4)
        f.truncate()

if __name__ == '__main__':
    migrate_knowledge_base()
