import os
import re
import shutil
import logging
import subprocess

from core.symphony.models import Workspace
from core.symphony.config import SymphonyConfig

logger = logging.getLogger(__name__)

class WorkspaceError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

class WorkspaceManager:
    def __init__(self, config: SymphonyConfig):
        self.config = config
        self.root = os.path.abspath(self.config.workspace_root)

    def _sanitize_key(self, identifier: str) -> str:
        """Replace non-alphanumeric/dot/underscore/dash with underscore."""
        return re.sub(r'[^A-Za-z0-9._-]', '_', identifier)

    def _get_workspace_path(self, identifier: str) -> str:
        key = self._sanitize_key(identifier)
        return os.path.join(self.root, key)

    def _run_hook(self, script: str, cwd: str, timeout_ms: int) -> None:
        if not script:
            return

        timeout_sec = timeout_ms / 1000.0
        try:
            # We use bash -lc to evaluate the script.
            subprocess.run(
                ['bash', '-lc', script],
                cwd=cwd,
                timeout=timeout_sec,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Hook timeout in {cwd}: {e}")
            raise WorkspaceError("hook_timeout", f"Hook execution timed out after {timeout_sec}s")
        except subprocess.CalledProcessError as e:
            logger.error(f"Hook failed in {cwd}: {e.stderr}")
            raise WorkspaceError("hook_failed", f"Hook execution failed with status {e.returncode}")

    def create_for_issue(self, identifier: str) -> Workspace:
        """Create or reuse workspace directory and run after_create hook if new."""
        key = self._sanitize_key(identifier)
        path = self._get_workspace_path(identifier)

        # Invariant 2: ensure path is under root
        if not os.path.abspath(path).startswith(self.root):
            raise WorkspaceError("invalid_workspace_path", "Workspace path must be within workspace root")

        created_now = False
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                created_now = True
            except Exception as e:
                raise WorkspaceError("workspace_creation_failed", f"Failed to create directory {path}: {e}")
        elif not os.path.isdir(path):
            raise WorkspaceError("invalid_workspace_path", f"Path exists but is not a directory: {path}")

        workspace = Workspace(path=path, workspace_key=key, created_now=created_now)

        if created_now and self.config.hook_after_create:
            try:
                self._run_hook(self.config.hook_after_create, path, self.config.hook_timeout_ms)
            except WorkspaceError:
                # Cleanup partially created workspace
                shutil.rmtree(path, ignore_errors=True)
                raise

        return workspace

    def run_before_run(self, path: str) -> None:
        if self.config.hook_before_run:
            self._run_hook(self.config.hook_before_run, path, self.config.hook_timeout_ms)

    def run_after_run(self, path: str) -> None:
        if self.config.hook_after_run:
            try:
                self._run_hook(self.config.hook_after_run, path, self.config.hook_timeout_ms)
            except Exception as e:
                logger.warning(f"Ignored after_run hook failure: {e}")

    def cleanup_workspace(self, identifier: str) -> None:
        path = self._get_workspace_path(identifier)
        if not os.path.exists(path):
            return

        if self.config.hook_before_remove:
            try:
                self._run_hook(self.config.hook_before_remove, path, self.config.hook_timeout_ms)
            except Exception as e:
                logger.warning(f"Ignored before_remove hook failure: {e}")

        try:
            shutil.rmtree(path)
        except Exception as e:
            logger.error(f"Failed to remove workspace {path}: {e}")
