"""
AFOS v30.1 — Zero-Trust OAuth2 Mailer (Replaces Legacy 16-Character Google App Password)
Complies with AFOS Step 5 P0 Security Checklist: Zero static credentials in code or local env.
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict, List

from google.auth.transport.requests import Request
from google. cloud import secretmanager
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, ConfigDict, EmailStr, Field

GMAIL_SEND_SCOPE: List[str] = ["https://www.googleapis.com/auth/gmail.send"]
JSONLOGIC_VERSION: str = "1.4.0"


class OutboundDispatchPayload(BaseModel):
    """Validated outbound notification contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sender: EmailStr
    recipient: EmailStr
    subject: str = Field(..., min_length=1, max_length=250)
    body_text: str = Field(..., min_length=1, max_length=100_000)
    secret_uri_ref: str = Field(
        default_factory=lambda: os.environ.get(
            "ADAM_GMAIL_OAUTH_SECRET_URI",
            "projects/adam-prod/secrets/gmail-smtp-oauth-token/versions/latest",
        )
    )


def _fetch_oauth_credentials_by_ref(secret_uri: str) -> Credentials:
    """Loads OAuth2 token payload strictly from Google Secret Manager by URI reference."""
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=secret_uri, timeout=10.0)
    secret_json = json.loads(response.payload.data.decode("utf-8"))

    creds = Credentials.from_authorized_user_info(secret_json, scopes=GMAIL_SEND_SCOPE)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
    return creds


def send_zero_trust_notification(
    payload: OutboundDispatchPayload,
    max_retries: int = 3,
    base_backoff_sec: float = 1.0,
) -> Dict[str, Any]:
    """
    Dispatches an email via the Gmail API using short-lived OAuth2 tokens and bounded jitter retries.
    Emits a W3C PROV-O audit trail entry upon completion.
    """
    creds = _fetch_oauth_credentials_by_ref(payload.secret_uri_ref)
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    mime_msg = EmailMessage()
    mime_msg.set_content(payload.body_text)
    mime_msg["To"] = str(payload.recipient)
    mime_msg["From"] = str(payload.sender)
    mime_msg["Subject"] = payload.subject

    encoded_msg = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("ascii")
    request_body = {"raw": encoded_msg}

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = (
                service.users()
                .messages()
                .send(userId="me", body=request_body)
                .execute(num_retries=0)
            )
            return {
                "status": "dispatched",
                "message_id": result.get("id"),
                "jsonLogic_version": JSONLOGIC_VERSION,
                "prov_o_audit_trail": [
                    {
                        "@type": "prov:Activity",
                        "prov:wasAssociatedWith": "urn:adam:agent:AFOS-SNT-01",
                        "prov:used": [f"urn:gcp:secretmanager:{payload.secret_uri_ref}"],
                        "prov:generated": f"urn:adam:notification:{result.get('id')}",
                        "prov:startedAtTime": datetime.now(timezone.utc).isoformat(),
                        "jsonLogic_version": JSONLOGIC_VERSION,
                        "confidence": 0.96,
                    }
                ],
            }
        except HttpError as exc:
            last_error = exc
            status_code = getattr(exc.resp, "status", 500)
            if status_code < 500 and status_code != 429:
                raise
            sleep_for = (base_backoff_sec * (2 ** (attempt - 1))) + random.uniform(0.05, 0.35)
            time.sleep(sleep_for)

    raise RuntimeError(f"Exhausted {max_retries} OAuth2 dispatch attempts: {last_error}")
