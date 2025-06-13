import json
import logging

def load_system_prompt(filepath: str) -> dict:
    """
    Loads a system prompt from a JSON file.
    Attempts to parse the entire file as a single JSON object.
    If that fails, it attempts to extract and parse the first valid JSON object
    found in the file, which can be useful for files with concatenated JSONs.
    """
    logger = logging.getLogger(__name__)
    # Ensure basicConfig is called if no handlers are configured for the root logger
    # This is important if this utility is used early or in contexts where logging isn't set up.
    if not logging.getLogger().hasHandlers() and not logging.getLogger(logger.name).hasHandlers():
        logging.basicConfig(level=logging.INFO) # Default to INFO if not configured

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Attempt to parse as a single JSON object first
        try:
            data = json.loads(content)
            logger.info(f"Successfully loaded system prompt from {filepath} as a single JSON object.")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse {filepath} as a single JSON object: {e}. Attempting to extract the first valid JSON object.")

            # Try to find the first complete JSON object (e.g., from '{' to matching '}')
            # This handles cases where multiple JSON documents might be concatenated or there's trailing text.
            first_json_content = None
            balance = 0
            start_index = -1
            first_json_end_index = -1

            for i, char in enumerate(content):
                if char == '{':
                    if start_index == -1:
                        start_index = i
                    balance += 1
                elif char == '}':
                    balance -= 1
                    if balance == 0 and start_index != -1:
                        first_json_end_index = i + 1
                        break # Found the first complete JSON object

            if start_index != -1 and first_json_end_index != -1:
                first_json_content = content[start_index:first_json_end_index]
                logger.info(f"Attempting to parse extracted first JSON object (approx. {len(first_json_content)} chars) from {filepath}.")
                try:
                    return json.loads(first_json_content)
                except json.JSONDecodeError as e_inner:
                    logger.error(f"Could not parse even the extracted first JSON object from {filepath}. Error: {e_inner}. Content snippet: {first_json_content[:200]}...")
                    return {}
            else:
                logger.error(f"Could not find a clearly delimited JSON object structure in {filepath}.")
                return {}

    except FileNotFoundError:
        logger.error(f"System prompt file not found: {filepath}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading system prompt from {filepath}: {e}", exc_info=True)
        return {}

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # Create a dummy prompt file for testing
    dummy_prompt_path = "dummy_prompt.txt"

    # Test 1: Valid JSON
    with open(dummy_prompt_path, 'w', encoding='utf-8') as f:
        json.dump({"name": "Adam", "version": "v19.2", "instructions_for_adam": ["Be helpful."]}, f)
    print(f"Test 1 (Valid JSON): {load_system_prompt(dummy_prompt_path)}")

    # Test 2: Concatenated JSONs (common issue)
    with open(dummy_prompt_path, 'w', encoding='utf-8') as f:
        f.write('{"name": "Adam First", "data": {"key": "value1"}}{"name": "Adam Second", "data": "othervalue"}')
    print(f"Test 2 (Concatenated JSONs): {load_system_prompt(dummy_prompt_path)}")

    # Test 3: JSON with trailing garbage
    with open(dummy_prompt_path, 'w', encoding='utf-8') as f:
        f.write('{"name": "Adam Valid", "info": "test"}trailing garbage text')
    print(f"Test 3 (JSON with trailing garbage): {load_system_prompt(dummy_prompt_path)}")

    # Test 4: Invalid JSON content
    with open(dummy_prompt_path, 'w', encoding='utf-8') as f:
        f.write('{"name": "Adam Invalid",, "info": "test"}')
    print(f"Test 4 (Invalid JSON): {load_system_prompt(dummy_prompt_path)}")

    # Test 5: File not found
    print(f"Test 5 (File not found): {load_system_prompt('non_existent_prompt.txt')}")

    # Clean up dummy file
    import os
    if os.path.exists(dummy_prompt_path):
        os.remove(dummy_prompt_path)
