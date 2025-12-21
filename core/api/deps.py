from core.engine.meta_orchestrator import MetaOrchestrator

_orchestrator = None


def get_orchestrator() -> MetaOrchestrator:
    """
    Returns a singleton instance of the MetaOrchestrator.
    """
    global _orchestrator
    if _orchestrator is None:
        # Initialize with default legacy orchestrator handled internally by MetaOrchestrator
        _orchestrator = MetaOrchestrator()
    return _orchestrator
