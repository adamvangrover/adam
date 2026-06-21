class CognitiveRouter:
    """
    Dynamically manages model routing for the Adam v30.1 Core Kernel.
    Gemini 1.5 Pro handles high-context synthesis (e.g., SEC Edgar, macro-physics);
    Claude 3.5 Sonnet handles meticulous formatting;
    GPT-4o handles high-speed RPC tool calling and rapid data extraction.
    """
    def route(self, task_type: str, context: str = "") -> str:
        task_type_lower = task_type.lower()
        if any(k in task_type_lower for k in ["high-context synthesis", "sec edgar", "macro-physics"]):
            return "Gemini 1.5 Pro"
        elif "meticulous formatting" in task_type_lower:
            return "Claude 3.5 Sonnet"
        elif any(k in task_type_lower for k in ["high-speed rpc tool calling", "rapid data extraction", "rpc"]):
            return "GPT-4o"
        return "GPT-4o"
