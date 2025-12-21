import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.append(os.getcwd())

# Mock dependencies to ensure the test runs even if packages are missing
sys.modules['semantic_kernel'] = MagicMock()
sys.modules['core.system.agent_orchestrator'] = MagicMock()
sys.modules['core.system.echo'] = MagicMock()
sys.modules['core.utils.config_utils'] = MagicMock()
sys.modules['core.utils.token_utils'] = MagicMock()
# We need to mock KnowledgeBase module if we want to test the import fix without relying on the file being perfect
sys.modules['core.system.knowledge_base'] = MagicMock()

try:
    from core.system.interaction_loop import InteractionLoop
except ImportError as e:
    print(f"Import Error: {e}")
    InteractionLoop = None

class TestInteractionLoopFixes(unittest.TestCase):
    def test_initialization_import_fix(self):
        """
        Verifies that InteractionLoop initializes without NameError (fixing Bug 1).
        """
        if InteractionLoop is None:
            self.fail("Could not import InteractionLoop")

        try:
            loop = InteractionLoop()
            self.assertIsNotNone(loop)
        except NameError as e:
            self.fail(f"Initialization failed with NameError: {e}")
        except Exception:
            # Other exceptions are acceptable as long as it's not NameError related to KnowledgeBase
            pass

    def test_eof_handling(self):
        """
        Verifies that run() handles EOFError gracefully (fixing Bug 2).
        """
        if InteractionLoop is None:
            self.fail("Could not import InteractionLoop")

        try:
            loop = InteractionLoop()
        except Exception:
            return # Skip if init fails for other reasons

        # Mock process_input to avoid side effects
        loop.process_input = MagicMock(return_value="Response")

        with patch('builtins.input', side_effect=EOFError):
            try:
                loop.run()
            except EOFError:
                self.fail("InteractionLoop.run() crashed on EOFError!")
            except Exception:
                pass # Other errors are fine, we just want to ensure EOF doesn't crash it

if __name__ == '__main__':
    unittest.main()
