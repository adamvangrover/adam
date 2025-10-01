# Prompt Library Guide

## 1. Introduction

This guide provides instructions for using and extending the ADAM v21.0 Prompt Library. The prompt library is a curated collection of prompts that are used by the AI agents to perform a variety of tasks, such as financial analysis, communication, and data extraction.

## 2. Using Existing Prompts

To use an existing prompt, you simply need to reference it by its module and name. For example, to use the `escalation_email` prompt from the `communication` module, you would reference it as `communication.escalation_email`.

The Agent Orchestrator will automatically load the prompts from the `prompt_library` directory and make them available to the agents.

## 3. Extending the Library

The prompt library is designed to be easily extensible. You can add new prompts or even create new modules for different tasks.

### 3.1. Creating a New Module

To create a new module, simply create a new subdirectory in the `prompt_library` directory. The name of the subdirectory will be the name of the module.

### 3.2. Creating a New Prompt

To create a new prompt, you need to create a JSON file in the appropriate module directory. The name of the file will be the name of the prompt. For example, to create a new prompt called `summarize_article` in a new `summarization` module, you would create the file `prompt_library/summarization/summarize_article.json`.

### 3.3. Prompt Formatting

Each prompt file should be a JSON object with the following properties:

*   `name`: The name of the prompt.
*   `description`: A brief description of what the prompt does.
*   `prompt`: The text of the prompt. You can use placeholders in the prompt text, which will be replaced with actual values at runtime. Placeholders should be enclosed in double curly braces (e.g., `{{article_text}}`). You can also use system-level placeholders, which will be automatically replaced by the system. For example, `{{current_date}}` will be replaced with the current date.

## 4. Best Practices for Prompt Engineering

*   **Be specific and clear:** The more specific you are in your prompt, the better the results will be. Avoid ambiguous language and provide as much context as possible.
*   **Use examples:** If you want the output to be in a specific format, provide an example in the prompt.
*   **Experiment and iterate:** Don't be afraid to experiment with different prompts to see what works best. The more you iterate, the better your prompts will become.
*   **Keep it simple:** Avoid overly complex prompts with too many instructions. It's often better to break down a complex task into smaller, simpler prompts.

## 5. Integrating Prompts into the System

Once you have created a new prompt, you need to integrate it into the system so that it can be used by the agents. This is done by referencing the prompt in a workflow definition file (e.g., `config/workflow21.yaml`).

For example, to use the `summarize_article` prompt in a workflow, you would add a step that calls an agent and passes the prompt name as a parameter.

## 6. Example: Creating a New "Summarization" Prompt

Let's walk through an example of creating a new prompt for summarizing an article.

**1. Create a new module:**

Create a new directory called `summarization` in the `prompt_library` directory.

**2. Create a new prompt file:**

Create a new file called `summarize_article.json` in the `prompt_library/summarization` directory with the following content:

```json
{
  "name": "summarize_article",
  "description": "Summarizes the key points of a given article.",
  "prompt": "Please summarize the following article in three key bullet points:\n\n{{article_text}}"
}
```

**3. Integrate the prompt into a workflow:**

Open your workflow YAML file and add a step that uses the new prompt. For example:

```yaml
- name: Summarize Article
  agent: TextProcessingAgent
  params:
    prompt: summarization.summarize_article
    inputs:
      article_text: "{{input_article}}"
```

Now, when this workflow is executed, the `TextProcessingAgent` will use the `summarize_article` prompt to summarize the input article.
