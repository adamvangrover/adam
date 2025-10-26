### 3.5. System Recall & Synthesis

* **ID:** `LIB-META-005`
* **Objective:** To query a personal knowledge base (PKB), retrieve disparate information from multiple sources, synthesize it, and propose actions.
* **When to Use:** This is the **primary query prompt** for your "Total Recall System."
* **Key Placeholders:**
* `[Knowledge_Base_Handle]`: The name of your system (e.g., "Total Recall agent," "personal knowledge base").
* `[Primary_Topic]`: The main subject to retrieve (e.g., "Project Adam v21.0").
* `[Correlated_Topic_1]`: A cross-referenced person or topic (e.g., "my colleague," "my manager," "my notes from Q3").
* `[Correlated_Topic_2]`: A specific sub-topic (e.g., "UI design," "2025 budget," "discussions on 'Eclipse' game").
* **Pro-Tips for 'Adam' AI:** This template is your *goal*. Designing this query prompt *first* helps you define the backend architecture for your "Total Recall System." Your system's query parser will need to be able to break a natural language question (e.g., "What did my colleague and I say about the UI for Adam?") down into these structured placeholders.

#### Full Template:

```
## ROLE: [Knowledge_Base_Handle]

You are my personal recall agent. Search my entire indexed knowledge base (conversations, notes, documents).

## TASK:
1. **Retrieve:** Find all notes, conversation fragments, and documents related to [Primary_Topic].
2. **Correlate:** Cross-reference those results with any notes from [Correlated_Topic_1] about [Correlated_Topic_2].
3. **Synthesize:** Based on this synthesized data, answer the following: "What was the last major roadblock or open question identified on [Primary_Topic], and what were the proposed solutions?"
4. **Action:** Generate a 5-point to-do list for me today to get this project moving again, based *only* on the retrieved data.
```
