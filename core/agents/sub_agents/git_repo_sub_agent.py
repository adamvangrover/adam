import subprocess
import os
from core.agents.agent_base import AgentBase


class GitRepoSubAgent(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.clone_dir = self.config.get("clone_dir", "downloads/repos")
        os.makedirs(self.clone_dir, exist_ok=True)

    def execute(self, repo_url: str, operation: str = "clone"):
        if operation == "clone":
            return self._clone_repo(repo_url)
        elif operation == "list_files":
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(self.clone_dir, repo_name)
            if not os.path.exists(repo_path):
                return {"error": "Repository not found. Please clone it first."}
            return self._list_files(repo_path)
        else:
            return {"error": f"Unsupported operation: {operation}"}

    def _clone_repo(self, repo_url: str):
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(self.clone_dir, repo_name)
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
