import json
import os
import glob
import re

def extract_agents():
    print("Extracting Agents...")
    agents = []
    base_path = 'core/agents'
    files = glob.glob(os.path.join(base_path, '*.py'))

    for file_path in files:
        if '__init__' in file_path:
            continue

        filename = os.path.basename(file_path)
        name = filename.replace('.py', '').replace('_', ' ').title()

        description = "No description available."
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Try to find class definition and docstring
                class_match = re.search(r'class\s+(\w+).*?:', content)
                if class_match:
                    name = class_match.group(1)

                # Simple docstring extraction (triple quotes)
                doc_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if doc_match:
                    description = doc_match.group(1).strip().replace('\n', ' ')[:200] + "..."
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        agents.append({
            "name": name,
            "filename": filename,
            "path": file_path,
            "description": description,
            "type": "System 2 Agent"
        })

    return agents

def extract_prompts():
    print("Extracting Prompts...")
    prompts = []
    base_path = 'prompt_library'
    # JSON files
    json_files = glob.glob(os.path.join(base_path, '*.json'))
    # MD files
    md_files = glob.glob(os.path.join(base_path, '*.md'))

    all_files = json_files + md_files

    for file_path in all_files:
        filename = os.path.basename(file_path)
        name = filename.split('.')[0].replace('_', ' ').title()
        format_type = filename.split('.')[-1].upper()

        content_preview = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_preview = content[:300] + "..."
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        prompts.append({
            "name": name,
            "filename": filename,
            "format": format_type,
            "preview": content_preview
        })

    return prompts

def extract_documentation_qa():
    print("Extracting Documentation QA from AGENTS.md...")
    qa_pairs = []
    try:
        with open('AGENTS.md', 'r', encoding='utf-8') as f:
            content = f.read()

        # Prime Directive
        if "The Prime Directive: Bifurcation" in content:
            qa_pairs.append({
                "messages": [
                    {"role": "user", "content": "What is the Prime Directive in the ADAM repository?"},
                    {"role": "assistant", "content": "The Prime Directive is 'Bifurcation'. It separates the codebase into two paths: 'Path A: The Product' (focused on reliability, located in core/agents) and 'Path B: The Lab' (focused on velocity, located in experimental/research)."}
                ]
            })

        # System 1 vs System 2
        if "Architecture: System 1 vs. System 2" in content:
            qa_pairs.append({
                "messages": [
                    {"role": "user", "content": "Explain the difference between System 1 and System 2 in ADAM."},
                    {"role": "assistant", "content": "System 1 (The Swarm) is asynchronous, follows a 'Fire-and-Forget' pattern, and is used for tasks like fetching data or monitoring. System 2 (The Graph) is synchronous, follows a State Machine pattern, and is used for complex reasoning and report generation."}
                ]
            })

        # Communication Protocols
        if "Communication Protocols" in content:
            qa_pairs.append({
                "messages": [
                    {"role": "user", "content": "What are the input and output schemas for System 2 agents?"},
                    {"role": "assistant", "content": "System 2 agents use `AgentInput` (query, context, tools) and `AgentOutput` (answer, sources, confidence, metadata). Confidence scores below 0.85 flag the result for review."}
                ]
            })

    except Exception as e:
        print(f"Error reading AGENTS.md: {e}")

    return qa_pairs

def extract_training_data(agents, prompts):
    print("Generating Synthetic Training Data...")
    training_data = []

    # 1. Add Documentation QA
    training_data.extend(extract_documentation_qa())

    # 2. Add Agent QA
    for agent in agents:
        training_data.append({
            "messages": [
                {"role": "user", "content": f"What is the role of the {agent['name']}?"},
                {"role": "assistant", "content": f"The {agent['name']} is a {agent['type']} in the ADAM system. Description: {agent['description']}"}
            ]
        })

    # 3. Add Prompt QA
    for prompt in prompts:
        training_data.append({
            "messages": [
                {"role": "user", "content": f"Show me the prompt template for {prompt['name']}."},
                {"role": "assistant", "content": f"Here is the {prompt['format']} prompt for {prompt['name']}:\n\n{prompt['preview']}"}
            ]
        })

    # 4. Add existing Tinker Lab data (as fallback/augmentation)
    input_path = 'tinker_lab/tinker-cookbook/example-data/conversations.jsonl'
    if os.path.exists(input_path):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 50: # Limit external samples to prioritize internal ones
                        break
                    try:
                        obj = json.loads(line)
                        training_data.append(obj)
                    except:
                        continue
        except Exception as e:
            print(f"Error reading external training data: {e}")

    return training_data

def main():
    output_dir = 'showcase/data'
    os.makedirs(output_dir, exist_ok=True)

    # Agents
    agents_data = extract_agents()
    with open(os.path.join(output_dir, 'seed_agents.json'), 'w') as f:
        json.dump(agents_data, f, indent=2)
    print(f"Saved {len(agents_data)} agents.")

    # Prompts
    prompts_data = extract_prompts()
    with open(os.path.join(output_dir, 'seed_prompts.json'), 'w') as f:
        json.dump(prompts_data, f, indent=2)
    print(f"Saved {len(prompts_data)} prompts.")

    # Training Data (Synthetic + External)
    training_data = extract_training_data(agents_data, prompts_data)
    with open(os.path.join(output_dir, 'seed_training_data.json'), 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved {len(training_data)} training samples.")

if __name__ == "__main__":
    main()
