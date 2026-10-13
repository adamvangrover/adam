import os
import subprocess
from typing import Dict, Any, List


def verify_artifact_at_commit(clean_ref: str, commit_sha: str) -> bool:
    """Verifies artifact existence either at a specific git commit or on local disk."""
    if commit_sha and commit_sha != "LOCAL":
        try:
            cmd = ["git", "cat-file", "-e", f"{commit_sha}:{clean_ref}"]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                return True
        except Exception:
            pass
    return os.path.exists(clean_ref)


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skill 1 — Epistemic State & Grounding Validator
    Extracts claims, verifies source references against actual repository snapshot/commit,
    records reason codes/gaps, and computes promotion_allowed deterministically by policy.
    """
    correlation_id = input_data.get("correlation_id", "unknown-correlation-id")
    commit_sha = input_data.get("commit_sha", "HEAD")
    assertions = input_data.get("assertions", [])

    validated_claims: List[Dict[str, Any]] = []
    all_verified = True if assertions else False

    for assertion in assertions:
        claim_id = assertion.get("claim_id", f"claim-{len(validated_claims)+1:03d}")
        text = assertion.get("text", "")
        source_refs = assertion.get("source_refs", [])

        reason_codes: List[str] = []
        evidence_refs: List[str] = []
        epistemic_status = "UNVERIFIED"
        verification_method = "repository_symbol_lookup"

        if not source_refs:
            reason_codes.append("NO_SUPPORTING_ARTIFACT")
            all_verified = False
        else:
            ref_valid = True
            for ref in source_refs:
                clean_ref = ref.strip().lstrip("/")
                if verify_artifact_at_commit(clean_ref, commit_sha):
                    evidence_refs.append(f"{clean_ref}@{commit_sha}:exists")
                else:
                    ref_valid = False
                    reason_codes.append(f"ARTIFACT_NOT_FOUND:{clean_ref}")

            if ref_valid and evidence_refs:
                epistemic_status = "VERIFIED"
                reason_codes.append("ARTIFACT_REFERENCE_VERIFIED")
            else:
                epistemic_status = "UNVERIFIED"
                all_verified = False

        validated_claims.append({
            "claim_id": claim_id,
            "text": text,
            "epistemic_status": epistemic_status,
            "evidence_refs": evidence_refs,
            "verification_method": verification_method,
            "reason_codes": reason_codes
        })

    # Deterministic policy calculation for promotion_allowed
    # Promotion is allowed iff there is at least one claim and all claims are VERIFIED
    promotion_allowed = all_verified and len(validated_claims) > 0

    status = "PASSED" if promotion_allowed else ("REVIEW_REQUIRED" if validated_claims else "REJECTED")

    return {
        "skill_id": "epistemic_grounding",
        "status": status,
        "claims": validated_claims,
        "promotion_allowed": promotion_allowed,
        "correlation_id": correlation_id
    }
