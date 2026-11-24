# Clean-Room Judge — publication edition

Score the edition's narrative (`summary` + `sections`) against its own structured data (the golden
answer is the engine-computed headline figure) on correctness, faithfulness to `sources`, reasoning
coherence and provenance citation. Use the weights given in `config/publication_review.yaml` → `judge.weights`.
Respond in the format `evals/graders/llm_judge.py::CleanRoomLLMJudge` parses:
one `<DIMENSION>: <0-1 score>` line per dimension, followed by `OVERALL_SCORE: <float>`.
