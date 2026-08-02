# AFOS Governance Kernel Specification

## Overview
The Governance Kernel is the immutable ledger of the operating system. It ensures that every decision made within AFOS is traceable, auditable, and compliant with institutional regulations.

## Core Components

1. **Policy Registry:** Maintains the authoritative repository of all historical and active policies.
2. **Execution Registry:** Tracks every workflow execution, mapping it to the specific agent, runtime, and timestamp.
3. **Version Registry:** Ensures strict version control for models, prompts, and deterministic logic.
4. **Provenance Tracking:** Implements W3C PROV-O standards to link every output to its source data and logic.
5. **Audit Ledger:** An append-only cryptographic log of all system actions.
6. **Replay Engine:** Facilitates the exact recreation of past decisions for regulatory audits.
7. **Human Review / Approval Chains:** Manages the routing of high-risk decisions (as flagged by the Policy Kernel) to human overseers via the Execution Kernel's interrupt mechanisms.

## The "Replayable Decision" Guarantee
The primary mandate of the Governance Kernel is to guarantee that an auditor can select any `Decision` from the ledger, retrieve the exact `Evidence` and `Policy` version from the Knowledge Kernel, and utilize the Replay Engine to arrive at the exact same conclusion.
