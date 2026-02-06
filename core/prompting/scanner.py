import os
import yaml
import re
from typing import List, Dict, Any
from core.prompting.loader import PROMPT_LIB_PATH, PromptLoader

class PromptScanner:
    """
    Scans the prompt library and returns metadata and scores for all prompts.
    """

    @staticmethod
    def scan(context: List[str] = None) -> List[Dict[str, Any]]:
        prompts = []

        # Extensions to look for
        valid_extensions = ('.md', '.json', '.yaml', '.yml', '.txt')

        for root, _, files in os.walk(PROMPT_LIB_PATH):
            for file in files:
                if file.endswith(valid_extensions):
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, PROMPT_LIB_PATH)
                    name = os.path.splitext(file)[0]

                    # Skip READMEs generally, unless specific ones are valuable
                    if name.upper() == "README":
                        continue

                    try:
                        prompt_data = PromptScanner._process_file(path, rel_path, name, context)
                        if prompt_data:
                            prompts.append(prompt_data)
                    except Exception as e:
                        # Log error but continue scanning
                        print(f"Error processing prompt {rel_path}: {e}")

        # Sort by score descending
        prompts.sort(key=lambda x: x.get('score', 0), reverse=True)
        return prompts

    @staticmethod
    def _process_file(path: str, rel_path: str, name: str, context: List[str] = None) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {}
        prompt_type = "TEXT"

        # Try to parse frontmatter if it's markdown
        if path.endswith('.md'):
            prompt_type = "MARKDOWN"
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        metadata = yaml.safe_load(parts[1]) or {}
                    except:
                        pass
        elif path.endswith('.json'):
            prompt_type = "JSON"
            try:
                data = yaml.safe_load(content) # JSON is valid YAML
                if isinstance(data, dict):
                    metadata = data.get('metadata', {})
            except:
                pass

        # Determine Category
        category = "General"
        if "AOPL" in rel_path:
            category = "AOPL (Official)"
        elif "system" in rel_path.lower():
            category = "System Architecture"
        elif "showcase" in rel_path.lower():
            category = "Showcase"
        elif "learning" in rel_path.lower():
            category = "Learning"

        # Extract Author/Version from metadata if available
        author = metadata.get('author', 'Unknown')
        version = metadata.get('version', '1.0')

        # Calculate Score
        score = PromptScanner._calculate_roi(content, metadata, rel_path, context)

        return {
            "id": rel_path.replace("/", "_").replace(".", "_"),
            "name": name,
            "path": rel_path,
            "category": category,
            "type": prompt_type,
            "content": content,
            "metadata": metadata,
            "author": author,
            "version": version,
            "score": score
        }

    @staticmethod
    def _calculate_roi(content: str, metadata: Dict[str, Any], rel_path: str, context: List[str] = None) -> int:
        score = 50 # Base Score

        # 1. Metadata Bonus
        if metadata:
            score += 10
            if metadata.get('author') == "Adam v23.5":
                score += 15 # High trust author

        # 2. Complexity/Length Bonus (up to 20)
        length_score = min(20, len(content) // 200)
        score += length_score

        # 3. Variable/Template Bonus (Reusable prompts are more valuable)
        # Count unique Jinja2 variables {{...}}
        variables = re.findall(r"\{\{\s*[\w_]+\s*\}\}", content)
        var_score = min(20, len(set(variables)) * 5)
        score += var_score

        # 4. Path/Category Bonus
        if "AOPL" in rel_path:
            score += 15
        if "system" in rel_path.lower():
            score += 10

        # 5. Structure Bonus (Headers)
        if "###" in content: # Markdown headers
            score += 5

        # 6. Context Boost
        if context:
            boost = 0
            search_text = (content + str(metadata) + rel_path).lower()
            for keyword in context:
                if keyword.lower() in search_text:
                    boost += 20
            score += min(40, boost) # Max 40 points boost

        # Cap at 100 (soft cap, can go slightly over with massive boost to indicate relevance)
        return min(120, score)
