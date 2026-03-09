import os
import yaml
from typing import Any, Dict, List, Optional
from core.symphony.models import WorkflowDefinition

class ConfigError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(self.message)

def expand_env_vars(val: Any) -> Any:
    """Expand $VAR or ${VAR} in strings."""
    if isinstance(val, str):
        return os.path.expandvars(val)
    elif isinstance(val, list):
        return [expand_env_vars(v) for v in val]
    elif isinstance(val, dict):
        return {k: expand_env_vars(v) for k, v in val.items()}
    return val

def expand_path(val: str) -> str:
    """Expand ~ and $VAR for paths."""
    if not val:
        return val
    # Expand env vars first, then expanduser
    expanded = os.path.expanduser(os.path.expandvars(val))
    return expanded

class SymphonyConfig:
    """Typed getters for workflow config values with environment indirection and defaults."""
    def __init__(self, raw_config: Dict[str, Any]):
        self._raw = expand_env_vars(raw_config) if raw_config else {}

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split('.')
        curr = self._raw
        for key in keys:
            if isinstance(curr, dict) and key in curr:
                curr = curr[key]
            else:
                return default
        return curr

    @property
    def tracker_kind(self) -> str:
        return self.get('tracker.kind', '')

    @property
    def tracker_endpoint(self) -> str:
        if self.tracker_kind == 'linear':
            return self.get('tracker.endpoint', 'https://api.linear.app/graphql')
        return self.get('tracker.endpoint', '')

    @property
    def tracker_api_key(self) -> str:
        return self.get('tracker.api_key', '')

    @property
    def tracker_project_slug(self) -> str:
        return self.get('tracker.project_slug', '')

    @property
    def tracker_active_states(self) -> List[str]:
        val = self.get('tracker.active_states', ['Todo', 'In Progress'])
        if isinstance(val, str):
            return [s.strip() for s in val.split(',')]
        return val

    @property
    def tracker_terminal_states(self) -> List[str]:
        val = self.get('tracker.terminal_states', ['Closed', 'Cancelled', 'Canceled', 'Duplicate', 'Done'])
        if isinstance(val, str):
            return [s.strip() for s in val.split(',')]
        return val

    @property
    def polling_interval_ms(self) -> int:
        return int(self.get('polling.interval_ms', 30000))

    @property
    def workspace_root(self) -> str:
        # Defaults to /tmp/symphony_workspaces or similar, assuming UNIX
        default = os.path.join(os.sep, 'tmp', 'symphony_workspaces')
        val = self.get('workspace.root', default)
        return expand_path(val)

    @property
    def hook_after_create(self) -> Optional[str]:
        return self.get('hooks.after_create')

    @property
    def hook_before_run(self) -> Optional[str]:
        return self.get('hooks.before_run')

    @property
    def hook_after_run(self) -> Optional[str]:
        return self.get('hooks.after_run')

    @property
    def hook_before_remove(self) -> Optional[str]:
        return self.get('hooks.before_remove')

    @property
    def hook_timeout_ms(self) -> int:
        val = int(self.get('hooks.timeout_ms', 60000))
        return val if val > 0 else 60000

    @property
    def max_concurrent_agents(self) -> int:
        return int(self.get('agent.max_concurrent_agents', 10))

    @property
    def max_turns(self) -> int:
        return int(self.get('agent.max_turns', 20))

    @property
    def max_retry_backoff_ms(self) -> int:
        return int(self.get('agent.max_retry_backoff_ms', 300000))

    @property
    def max_concurrent_agents_by_state(self) -> Dict[str, int]:
        raw_map = self.get('agent.max_concurrent_agents_by_state', {})
        if not isinstance(raw_map, dict):
            return {}
        normalized = {}
        for k, v in raw_map.items():
            try:
                val = int(v)
                if val > 0:
                    normalized[k.strip().lower()] = val
            except (ValueError, TypeError):
                pass
        return normalized

    @property
    def codex_command(self) -> str:
        return self.get('codex.command', 'codex app-server')

    @property
    def codex_approval_policy(self) -> Optional[str]:
        return self.get('codex.approval_policy')

    @property
    def codex_thread_sandbox(self) -> Optional[str]:
        return self.get('codex.thread_sandbox')

    @property
    def codex_turn_sandbox_policy(self) -> Optional[Dict[str, Any]]:
        return self.get('codex.turn_sandbox_policy')

    @property
    def codex_turn_timeout_ms(self) -> int:
        return int(self.get('codex.turn_timeout_ms', 3600000))

    @property
    def codex_read_timeout_ms(self) -> int:
        return int(self.get('codex.read_timeout_ms', 5000))

    @property
    def codex_stall_timeout_ms(self) -> int:
        return int(self.get('codex.stall_timeout_ms', 300000))

    @property
    def server_port(self) -> Optional[int]:
        port = self.get('server.port')
        return int(port) if port is not None else None


def load_workflow(filepath: str) -> WorkflowDefinition:
    """Reads WORKFLOW.md, parses YAML front matter and prompt body."""
    if not os.path.exists(filepath):
        raise ConfigError('missing_workflow_file', f"Workflow file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise ConfigError('missing_workflow_file', f"Could not read {filepath}: {e}")

    config = {}
    prompt_template = content.strip()

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_str = parts[1]
            try:
                config = yaml.safe_load(yaml_str) or {}
            except yaml.YAMLError as e:
                raise ConfigError('workflow_parse_error', f"Invalid YAML front matter: {e}")

            if not isinstance(config, dict):
                raise ConfigError('workflow_front_matter_not_a_map', "YAML front matter must be an object/map")

            prompt_template = parts[2].strip()

    return WorkflowDefinition(config=config, prompt_template=prompt_template)

def validate_config(config: SymphonyConfig) -> Optional[ConfigError]:
    """Preflight validation to check if the orchestrator can run this config."""
    if not config.tracker_kind:
        return ConfigError('invalid_config', 'tracker.kind is missing')
    if config.tracker_kind not in ['linear']:
        return ConfigError('invalid_config', f"unsupported tracker.kind: {config.tracker_kind}")
    if not config.tracker_api_key:
        return ConfigError('invalid_config', 'tracker.api_key is missing or empty after resolution')
    if config.tracker_kind == 'linear' and not config.tracker_project_slug:
        return ConfigError('invalid_config', 'tracker.project_slug is required for linear tracker')
    if not config.codex_command:
        return ConfigError('invalid_config', 'codex.command is missing')
    return None
