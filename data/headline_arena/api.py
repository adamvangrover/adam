"""
ADAM-Macro-Sentinel API Client
===============================
Headline Arena API v1 client with token lifecycle management.
Wraps registration, authentication, challenge discovery, and prediction
submission with retry logic and audit logging.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests


class HeadlineArenaClient:
    """
    HTTP client for the Headline Arena evaluation platform.
    Manages the full agent lifecycle:
        Register → Challenge → Auth → Subscribe → Discover → Predict
    """

    BASE_URL = "https://headlinearena.com"

    AGENT_CONFIG = {
        "name": "ADAM-Macro-Sentinel",
        "type": "commenter",
        "bio": (
            "Institutional-grade macro forecasting engine from the ADAM Financial "
            "Operating System. Synthesizes central bank policy transmission, "
            "commodity supply chains, geopolitical risk premia, cross-asset flow "
            "dynamics, and labour-market telemetry into calibrated directional "
            "market calls with structured 4-dimension epistemic rationales."
        ),
        "languages": ["en"],
        "model_provider": "Anthropic",
        "model_name": "claude-opus-4-6",
        "auth_method": "client_credentials",
        "requested_scopes": [
            "prediction:submit",
            "challenge:read",
            "comment:create",
            "comment:reply",
        ],
    }

    def __init__(self, creds_path: str | Path):
        self.creds_path = Path(creds_path)
        self._creds: dict = {}
        self._token: str = ""
        self._load_creds()

    def _load_creds(self) -> None:
        if self.creds_path.exists():
            self._creds = json.loads(self.creds_path.read_text())
            self._token = self._creds.get("access_token", "")

    def _save_creds(self) -> None:
        self.creds_path.write_text(json.dumps(self._creds, indent=2))

    # ─── HTTP primitives ─────────────────────────────────────────────────

    def _post(
        self,
        path: str,
        json_data: dict | None = None,
        token: str | None = None,
        retries: int = 2,
    ) -> dict:
        headers = {"Content-Type": "application/json"}
        tok = token or self._token
        if tok:
            headers["Authorization"] = f"Bearer {tok}"

        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    f"{self.BASE_URL}{path}",
                    json=json_data,
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 5))
                    print(f"  ⏳ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500 and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                return resp.json()
            except requests.exceptions.Timeout:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": "timeout", "status_code": 0}
            except Exception as e:
                return {"error": str(e), "status_code": 0}

        return {"error": "max retries exceeded"}

    def _get(self, path: str, token: str | None = None) -> dict:
        headers = {}
        tok = token or self._token
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        try:
            resp = requests.get(
                f"{self.BASE_URL}{path}",
                headers=headers,
                timeout=30,
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ─── Agent lifecycle ─────────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return self._creds.get("agent_id", "")

    @property
    def token(self) -> str:
        return self._token

    @property
    def is_registered(self) -> bool:
        return bool(self._creds.get("agent_id"))

    def register(self) -> dict:
        """Register the agent if not already registered."""
        if self.is_registered:
            return self._creds

        result = self._post(
            "/api/v1/agent/registry/register",
            self.AGENT_CONFIG,
        )

        if "agent_id" in result:
            self._creds = {
                "agent_id": result["agent_id"],
                "client_secret": result.get("client_secret", ""),
                "challenge_id": result.get("challenge_id", ""),
                "challenge_prompt": result.get("challenge_prompt", ""),
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_creds()

        return result

    def authenticate(self) -> str:
        """Obtain a fresh access token via client_credentials flow."""
        result = self._post(
            "/api/v1/agent/auth/token",
            {
                "grant_type": "client_credentials",
                "agent_id": self._creds.get("agent_id", ""),
                "client_secret": self._creds.get("client_secret", ""),
            },
        )

        token = result.get("access_token", "")
        if token:
            self._token = token
            self._creds["access_token"] = token
            self._creds["token_obtained_at"] = datetime.now(timezone.utc).isoformat()
            self._save_creds()

        return token

    def discover_challenges(self) -> list[dict]:
        """Discover all active challenges."""
        result = self._get("/api/v1/eval/challenges/active")
        return result.get("challenges", [])

    def submit_prediction(
        self,
        challenge_id: str,
        payload: dict,
    ) -> dict:
        """Submit a single prediction to a challenge."""
        return self._post(
            f"/api/v1/eval/challenges/{challenge_id}/predict",
            payload,
        )

    def submit_batch(
        self,
        predictions: list[dict],
    ) -> list[dict]:
        """Submit multiple predictions, returning results."""
        results = []
        for pred in predictions:
            cid = pred.get("challenge_id", "")
            result = self.submit_prediction(cid, pred)
            results.append({
                "challenge_id": cid,
                "response": result,
                "scored": result.get("counts_for_score", False),
            })
        return results

    def get_agent_profile(self) -> dict:
        """Fetch the agent's public profile."""
        return self._get(f"/api/v1/agent/profile/{self.agent_id}")

    def get_rankings(self) -> dict:
        """Fetch current rankings."""
        return self._get("/api/v1/eval/rankings")
