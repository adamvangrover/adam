import json
import jsonschema

class SchemaMiddleware:
    """
    Middleware that deterministicly checks JSON LLM output.
    If parsing fails or schema validation fails, it intercepts the output and constructs a new prompt
    looping the error back to the LLM.
    """
    def __init__(self, schema: dict = None):
        self.schema = schema
        self.validator = None
        if schema:
                # Bolt Optimization: Pre-compile jsonschema validator to avoid recompilation overhead on every validation request.
            ValidatorClass = jsonschema.validators.validator_for(schema)
            self.validator = ValidatorClass(schema)

    def validate_and_parse(self, llm_output: str):
        """
        Validates the raw LLM string output against the schema.
        Returns the parsed JSON if valid.
        Returns a constructed retry prompt string if invalid.
        """
        try:
            parsed = json.loads(llm_output)

            if self.schema:
                self.validator.validate(instance=parsed)

            return True, parsed
        except json.JSONDecodeError as e:
            retry_prompt = f"You failed schema validation with this error: [{str(e)}]. Fix it."
            return False, retry_prompt
        except jsonschema.exceptions.ValidationError as e:
            retry_prompt = f"You failed schema validation with this error: [{str(e)}]. Fix it."
            return False, retry_prompt
