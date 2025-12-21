import subprocess
import os
from core.agents.agent_base import AgentBase

class GitRepoSubAgent(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.clone_dir = self.config.get("clone_dir", "downloads/repos")
        os.makedirs(self.clone_dir, exist_ok=True)

    async def execute(self, repo_url: str, operation: str = "clone"):
        try:
            self._validate_url(repo_url)
            repo_path = self._get_safe_repo_path(repo_url)
        except ValueError as e:
            return {"error": str(e)}

        if operation == "clone":
            return self._clone_repo(repo_url, repo_path)
        elif operation == "list_files":
            if not os.path.exists(repo_path):
                return {"error": "Repository not found. Please clone it first."}
            return self._list_files(repo_path)
        else:
            return {"error": f"Unsupported operation: {operation}"}

    def _validate_url(self, repo_url: str):
        if not repo_url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL scheme. Only http/https are allowed.")
        if repo_url.startswith("-"):
            raise ValueError("Invalid URL: cannot start with '-'")

    def _get_safe_repo_path(self, repo_url: str):
        # Extract name and prevent simple traversal characters in the name itself
        repo_name = repo_url.rstrip('/').split("/")[-1].replace(".git", "")

        if not repo_name:
             raise ValueError("Could not determine repository name from URL.")

        if ".." in repo_name or "/" in repo_name or "\\" in repo_name:
             raise ValueError("Invalid repository name extracted from URL.")

        repo_path = os.path.join(self.clone_dir, repo_name)
        abs_repo_path = os.path.abspath(repo_path)
        abs_clone_dir = os.path.abspath(self.clone_dir)

        # Ensure the resolved path is strictly inside the clone directory
        if not abs_repo_path.startswith(abs_clone_dir) or abs_repo_path == abs_clone_dir:
             raise ValueError("Security Violation: Path traversal detected.")

        return abs_repo_path

    def _clone_repo(self, repo_url: str, repo_path: str):
        try:
            if os.path.exists(repo_path):
                return {"status": "already exists", "path": repo_path}

            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            return {"status": "cloned", "path": repo_path}
        except subprocess.CalledProcessError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    def _list_files(self, repo_path: str):
        files = []
        for root, _, filenames in os.walk(repo_path):
            for filename in filenames:
                files.append(os.path.join(root, filename).replace(repo_path, "", 1))
        return {"files": files}