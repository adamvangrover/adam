import json

class Kernel:
    def log_state(self, state):
        with open("state.jsonl", "a") as f:
            f.write(json.dumps(state) + "\n")
