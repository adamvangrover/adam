import pytest
from unittest.mock import patch
from core.symphony.config import SymphonyConfig
from core.symphony.tracker import LinearTrackerClient, TrackerError

@pytest.fixture
def config():
    return SymphonyConfig({
        "tracker": {
            "kind": "linear",
            "api_key": "dummy",
            "project_slug": "ENG"
        }
    })

def test_tracker_initialization(config):
    client = LinearTrackerClient(config)
    assert client.endpoint == "https://api.linear.app/graphql"
    assert client.api_key == "dummy"
    assert client.project_slug == "ENG"

def test_tracker_missing_config():
    with pytest.raises(TrackerError):
        LinearTrackerClient(SymphonyConfig({"tracker": {"kind": "linear", "api_key": ""}}))

@patch.object(LinearTrackerClient, '_post')
def test_fetch_candidate_issues(mock_post, config):
    mock_post.return_value = {
        "issues": {
            "pageInfo": {"hasNextPage": False},
            "nodes": [
                {
                    "id": "abc1",
                    "identifier": "ENG-1",
                    "title": "Task 1",
                    "state": {"name": "Todo"},
                    "labels": {"nodes": [{"name": "Bug"}]},
                    "priority": 1
                }
            ]
        }
    }

    client = LinearTrackerClient(config)
    issues = client.fetch_candidate_issues()

    assert len(issues) == 1
    assert issues[0].id == "abc1"
    assert issues[0].identifier == "ENG-1"
    assert issues[0].state == "Todo"
    assert issues[0].labels == ["bug"]
    assert issues[0].priority == 1

    mock_post.assert_called_once()
    variables = mock_post.call_args[0][1]
    assert "project" in variables["filter"]
    assert "slugId" in variables["filter"]["project"]

@patch.object(LinearTrackerClient, '_post')
def test_fetch_paginated_issues(mock_post, config):
    page1 = {
        "issues": {
            "pageInfo": {"hasNextPage": True, "endCursor": "cursor1"},
            "nodes": [{"id": "abc1", "identifier": "ENG-1", "title": "1", "state": {"name": "Todo"}}]
        }
    }
    page2 = {
        "issues": {
            "pageInfo": {"hasNextPage": False},
            "nodes": [{"id": "abc2", "identifier": "ENG-2", "title": "2", "state": {"name": "Todo"}}]
        }
    }
    mock_post.side_effect = [page1, page2]

    client = LinearTrackerClient(config)
    issues = client.fetch_candidate_issues()

    assert len(issues) == 2
    assert mock_post.call_count == 2

@patch.object(LinearTrackerClient, '_post')
def test_blocker_normalization(mock_post, config):
    mock_post.return_value = {
        "issues": {
            "pageInfo": {"hasNextPage": False},
            "nodes": [
                {
                    "id": "abc1",
                    "identifier": "ENG-1",
                    "title": "Task 1",
                    "state": {"name": "Todo"},
                    "inverseRelations": {
                        "nodes": [
                            {
                                "type": "blocks",
                                "issue": {
                                    "id": "def456",
                                    "identifier": "ENG-2",
                                    "state": {"name": "In Progress"}
                                }
                            }
                        ]
                    }
                }
            ]
        }
    }

    client = LinearTrackerClient(config)
    issues = client.fetch_candidate_issues()
    assert len(issues[0].blocked_by) == 1
    assert issues[0].blocked_by[0].id == "def456"
    assert issues[0].blocked_by[0].state == "In Progress"
