import os
from openai import OpenAI


class ArchitectAgent:
    """
    The Architect Agent is responsible for maintaining, optimizing, and evolving
    the system infrastructure and reasoning logic.
    """

    def __init__(self):
        with open("core/agents/architect_agent/prompts/system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.client = OpenAI(api_key=api_key)

    def run(self, user_query):
        """
        This is the main loop for the Architect Agent.
        """
        print("Architect Agent is running.")

        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        print(completion.choices[0].message.content)


if __name__ == "__main__":
    # This agent will now raise a ValueError if the OPENAI_API_KEY is not set.
    # To run this, you would need to set the environment variable first.
    # For example:
    # export OPENAI_API_KEY="your-api-key"
    # python core/agents/architect_agent/agent.py

    # Since we are in a non-interactive environment and cannot set the key,
    # we will not run the agent here. The code is for demonstration purposes.
    pass
