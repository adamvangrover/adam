import logging
from typing import Any, Dict, List, Optional
import httpx

from core.symphony.models import Issue, BlockerRef
from core.symphony.config import SymphonyConfig

logger = logging.getLogger(__name__)

class TrackerError(Exception):
    def __init__(self, code: str, message: str, payload: Optional[Any] = None):
        self.code = code
        self.message = message
        self.payload = payload
        super().__init__(self.message)

class TrackerClient:
    """Abstract tracker client."""
    def fetch_candidate_issues(self) -> List[Issue]:
        raise NotImplementedError

    def fetch_issues_by_states(self, state_names: List[str]) -> List[Issue]:
        raise NotImplementedError

    def fetch_issue_states_by_ids(self, issue_ids: List[str]) -> List[Issue]:
        raise NotImplementedError

class LinearTrackerClient(TrackerClient):
    """Linear GraphQL implementation of the tracker client."""
    def __init__(self, config: SymphonyConfig):
        self.endpoint = config.tracker_endpoint
        self.api_key = config.tracker_api_key
        self.project_slug = config.tracker_project_slug
        self.active_states = config.tracker_active_states

        if not self.api_key:
            raise TrackerError('missing_tracker_api_key', 'Linear API key is missing')
        if not self.project_slug:
            raise TrackerError('missing_tracker_project_slug', 'Linear project slug is missing')

    def _post(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                    headers=headers
                )

            if resp.status_code != 200:
                raise TrackerError('linear_api_status', f"Linear API returned {resp.status_code}", payload=resp.text)

            data = resp.json()
            if "errors" in data:
                raise TrackerError('linear_graphql_errors', "Linear GraphQL errors", payload=data["errors"])
            return data.get("data", {})
        except httpx.RequestError as e:
            raise TrackerError('linear_api_request', f"Linear API request failed: {e}")
        except Exception as e:
            if isinstance(e, TrackerError):
                raise
            raise TrackerError('linear_unknown_payload', f"Failed to parse Linear response: {e}")

    def _normalize_issue(self, node: Dict[str, Any]) -> Issue:
        state_name = node.get("state", {}).get("name", "")
        labels = [l.get("name", "").lower() for l in node.get("labels", {}).get("nodes", [])]

        blockers = []
        # In Linear, blocked by relation is typically an inverse relation "blocks"
        for rel in node.get("inverseRelations", {}).get("nodes", []):
            if rel.get("type") == "blocks" and rel.get("issue"):
                related = rel["issue"]
                blockers.append(BlockerRef(
                    id=related.get("id"),
                    identifier=related.get("identifier"),
                    state=related.get("state", {}).get("name")
                ))

        priority = node.get("priority")
        if not isinstance(priority, int):
            priority = None

        from datetime import datetime
        created_at = None
        if node.get("createdAt"):
            try:
                # Handle ISO formatting with Z
                created_at = datetime.fromisoformat(node["createdAt"].replace('Z', '+00:00'))
            except ValueError:
                pass

        updated_at = None
        if node.get("updatedAt"):
            try:
                updated_at = datetime.fromisoformat(node["updatedAt"].replace('Z', '+00:00'))
            except ValueError:
                pass

        return Issue(
            id=node["id"],
            identifier=node["identifier"],
            title=node["title"],
            description=node.get("description"),
            priority=priority,
            state=state_name,
            branch_name=node.get("branchName"),
            url=node.get("url"),
            labels=labels,
            blocked_by=blockers,
            created_at=created_at,
            updated_at=updated_at
        )

    def _fetch_paginated_issues(self, filter_obj: Dict[str, Any]) -> List[Issue]:
        query = """
        query GetIssues($filter: IssueFilter, $after: String) {
          issues(first: 50, filter: $filter, after: $after) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              id
              identifier
              title
              description
              priority
              branchName
              url
              createdAt
              updatedAt
              state {
                name
              }
              labels {
                nodes {
                  name
                }
              }
              inverseRelations {
                nodes {
                  type
                  issue {
                    id
                    identifier
                    state {
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """
        issues = []
        has_next = True
        cursor = None

        while has_next:
            variables = {"filter": filter_obj, "after": cursor}
            data = self._post(query, variables)

            issues_conn = data.get("issues", {})
            nodes = issues_conn.get("nodes", [])
            for node in nodes:
                try:
                    issues.append(self._normalize_issue(node))
                except Exception as e:
                    logger.warning(f"Failed to normalize issue {node.get('identifier')}: {e}")

            page_info = issues_conn.get("pageInfo", {})
            has_next = page_info.get("hasNextPage", False)
            if has_next:
                cursor = page_info.get("endCursor")
                if not cursor:
                    raise TrackerError('linear_missing_end_cursor', "Pagination error: missing endCursor")
        return issues

    def fetch_candidate_issues(self) -> List[Issue]:
        """Return issues in configured active states for a configured project."""
        filter_obj = {
            "project": {"slugId": {"eq": self.project_slug}},
            "state": {"name": {"in": self.active_states}}
        }
        return self._fetch_paginated_issues(filter_obj)

    def fetch_issues_by_states(self, state_names: List[str]) -> List[Issue]:
        """Used for startup terminal cleanup."""
        if not state_names:
            return []
        filter_obj = {
            "project": {"slugId": {"eq": self.project_slug}},
            "state": {"name": {"in": state_names}}
        }
        return self._fetch_paginated_issues(filter_obj)

    def fetch_issue_states_by_ids(self, issue_ids: List[str]) -> List[Issue]:
        """Used for active-run reconciliation."""
        if not issue_ids:
            return []
        query = """
        query GetIssuesById($ids: [ID!]!) {
          issues(filter: { id: { in: $ids } }) {
            nodes {
              id
              identifier
              title
              description
              priority
              branchName
              url
              createdAt
              updatedAt
              state {
                name
              }
              labels {
                nodes {
                  name
                }
              }
              inverseRelations {
                nodes {
                  type
                  issue {
                    id
                    identifier
                    state {
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """
        data = self._post(query, {"ids": issue_ids})
        nodes = data.get("issues", {}).get("nodes", [])
        return [self._normalize_issue(node) for node in nodes]


def create_tracker_client(config: SymphonyConfig) -> TrackerClient:
    kind = config.tracker_kind
    if kind == 'linear':
        return LinearTrackerClient(config)
    raise TrackerError('unsupported_tracker_kind', f"Unsupported tracker.kind: {kind}")
