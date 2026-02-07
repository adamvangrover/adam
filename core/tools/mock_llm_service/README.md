# Mock LLM Service for Adam Chatbot Simulation

This service simulates an LLM API endpoint for development and testing of the Adam system, particularly the command-line chatbot (`scripts/run_chatbot.py`). It uses a rule-based "Reference Probability Map" to generate contextual responses.

## Setup

1.  Navigate to this directory:
    ```bash
    cd tools/mock_llm_service
    ```
2.  Create a Python virtual environment:
    ```bash
    python -m venv venv
    ```
3.  Activate the virtual environment:
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Service

Ensure your virtual environment is activated. Then, run the Flask application:
```bash
python app.py
```
The service will start by default on `http://localhost:5001`. You should see output indicating it's running, similar to:
`* Running on http://127.0.0.1:5001/ (Press CTRL+C to quit)`

## Endpoints

*   **POST `/mock_complete`**:
    *   This is the primary endpoint that simulates LLM completions.
    *   It expects a JSON payload with a "prompt" key: `{"prompt": "Your prompt text here..."}`
    *   It returns a JSON response in the format: `{"choices": [{"message": {"role": "assistant", "content": "Simulated response..."}}]}`
