package adam_os.authz

import future.keywords.in

default allow := false

# Role-based access for the Sovereign Analyst
allow {
    input.role == "sovereign_analyst"
    input.action in ["evaluate_risk", "execute_lbo", "read_telemetry"]
}

# Deny any action flagged as non-deterministic
deny {
    input.action == "execute_non_deterministic_math"
}
