# Prompts

This directory contains prompts for interacting with the large language model (LLM) in the ADAM system. Prompts are used to guide the LLM in generating text, answering questions, and performing other natural language processing tasks.

## Prompt Format

Prompts are stored in JSON format. Each prompt has the following structure:

```json
{
  "name": "prompt_name",
  "description": "A brief description of the prompt.",
  "prompt": "The text of the prompt."
}
```

*   **`name`:** A unique name for the prompt.
*   **`description`:** A brief description of what the prompt does.
*   **`prompt`:** The text of the prompt. This can include placeholders that will be replaced with dynamic values at runtime.

## Prompt Engineering Best Practices

To get the best results from the LLM, it is important to follow these best practices for prompt engineering:

### Be Specific and Clear

The more specific and clear you are in your prompt, the better the LLM will be able to understand what you are asking for. Avoid ambiguous language and provide as much context as possible.

**Good Example:**

> "Generate a summary of the following news article, focusing on the financial implications for Apple Inc."

**Bad Example:**

> "Summarize this article."

### Provide Examples

Providing examples in your prompt can help the LLM to understand the format and style of the desired output. This is especially useful for tasks such as text generation and code generation.

**Good Example:**

> "Generate a Python function that takes two numbers as input and returns their sum. For example, if the input is `(2, 3)`, the output should be `5`."

**Bad Example:**

> "Write a Python function to add two numbers."

### Use Placeholders

Using placeholders in your prompts can make them more reusable and adaptable to different situations. This is especially useful for prompts that are used in automated workflows.

**Good Example:**

> "Generate a report on the financial performance of {{company_name}} for the last quarter."

**Bad Example:**

> "Generate a report on the financial performance of Apple for the last quarter."

### Iterate and Refine

Writing good prompts is an iterative process. Don't be afraid to experiment with different phrasings and formats to see what works best. The `prompt_tuner` agent can be used to help you refine your prompts.

## Using Prompts

To use a prompt, you will need to load the prompt from the JSON file and then pass it to the LLM engine. The LLM engine will then replace any placeholders in the prompt with the values you provide and generate a response.

## Creating New Prompts

When creating new prompts, please follow these guidelines:

*   **Be specific.** The more specific the prompt, the better the results will be.
*   **Use placeholders.** Use placeholders to make your prompts more reusable.
*   **Test your prompts.** Test your prompts with a variety of inputs to ensure that they are working as expected.

## Best Practices

For more information on best practices for writing prompts, please refer to the `PROMPT_BEST_PRACTICES.md` file in this directory.

By following these guidelines, you can help to ensure that the prompts used in the ADAM system are effective, reusable, and easy to maintain.
