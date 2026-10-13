export type ArbitrationOutcome =
  | "ADMIT_PROPOSAL"
  | "REJECT"
  | "QUARANTINE"
  | "ESCALATE"
  | "ABSTAIN";

export interface ModelAssessment {
  modelId: string;
  modelVersion: string;
  reasoningPathHash: string;
  evidenceRefs: string[];
  claimValues: Record<string, number | string | boolean>;
  confidenceScore: number;
}

export interface ArbitrationProposal {
  proposalId: string;
  proposalHash: string;
  correlationId: string;
  policyVersion: string;
  assessments: ModelAssessment[];
  requiredCapabilities: string[];
  grantedCapabilities: string[];
}

export interface ArbitrationReceipt {
  schemaVersion: "1.0";
  correlationId: string;
  proposalHash: string;
  policyVersion: string;
  modelAssessmentRefs: string[];
  evidenceRefs: string[];
  outcome: ArbitrationOutcome;
  reasonCodes: string[];
  admissionReceiptRef: string | null;
}

export class DualModelArbitrationGate {
  /**
   * Compares independent model assessments and enforces policy gates.
   * ADMIT_PROPOSAL means the proposal passed the arbitration gate.
   * It does NOT grant authoritative state; final admission remains with governed state machine.
   */
  public evaluateProposal(proposal: ArbitrationProposal): ArbitrationReceipt {
    const reasonCodes: string[] = [];
    const modelRefs = proposal.assessments.map((a) => `${a.modelId}@${a.modelVersion}`);
    const evidenceSet = new Set<string>();
    proposal.assessments.forEach((a) => a.evidenceRefs.forEach((e) => evidenceSet.add(e)));
    const evidenceRefs = Array.from(evidenceSet);

    // 1. Evidence Completeness Check
    if (evidenceRefs.length === 0) {
      reasonCodes.push("MISSING_EVIDENCE_REFERENCES");
      return this.buildReceipt(proposal, "REJECT", modelRefs, evidenceRefs, reasonCodes);
    }

    // 2. Minimum Assessment Requirement (Multi-Model Check)
    if (proposal.assessments.length < 2) {
      reasonCodes.push("SINGLE_MODEL_INSUFFICIENT_FOR_ARBITRATION");
      return this.buildReceipt(proposal, "ABSTAIN", modelRefs, evidenceRefs, reasonCodes);
    }

    // 3. Independence Check: Pairwise check for correlated reasoning paths
    for (let i = 0; i < proposal.assessments.length; i++) {
      for (let j = i + 1; j < proposal.assessments.length; j++) {
        if (proposal.assessments[i].reasoningPathHash === proposal.assessments[j].reasoningPathHash) {
          reasonCodes.push("CORRELATED_REASONING_PATH_FAILURE");
          return this.buildReceipt(proposal, "QUARANTINE", modelRefs, evidenceRefs, reasonCodes);
        }
      }
    }

    // 4. Capability & Scope Compliance
    const missingCaps = proposal.requiredCapabilities.filter(
      (cap) => !proposal.grantedCapabilities.includes(cap)
    );
    if (missingCaps.length > 0) {
      reasonCodes.push("UNAUTHORIZED_CAPABILITY_SCOPE");
      return this.buildReceipt(proposal, "REJECT", modelRefs, evidenceRefs, reasonCodes);
    }

    // 5. Material Claims Agreement Check across all model assessment pairs
    let materialConflict = false;
    for (let i = 0; i < proposal.assessments.length; i++) {
      for (let j = i + 1; j < proposal.assessments.length; j++) {
        const m1 = proposal.assessments[i];
        const m2 = proposal.assessments[j];
        const allKeys = new Set([...Object.keys(m1.claimValues), ...Object.keys(m2.claimValues)]);
        for (const key of allKeys) {
          const val1 = m1.claimValues[key];
          const val2 = m2.claimValues[key];
          if (val1 !== undefined && val2 !== undefined && val1 !== val2) {
            materialConflict = true;
            reasonCodes.push(`MATERIAL_DISAGREEMENT_ON_${key.toUpperCase()}`);
          }
        }
      }
    }

    if (materialConflict) {
      reasonCodes.push("DISAGREEMENT_ACROSS_MODEL_ASSESSMENTS");
      return this.buildReceipt(proposal, "ESCALATE", modelRefs, evidenceRefs, reasonCodes);
    }

    // 6. Passed all arbitration checks -> ADMIT_PROPOSAL
    reasonCodes.push("DUAL_MODEL_CONVERGENCE_SATISFIED");
    return this.buildReceipt(proposal, "ADMIT_PROPOSAL", modelRefs, evidenceRefs, reasonCodes, null);
  }

  private buildReceipt(
    proposal: ArbitrationProposal,
    outcome: ArbitrationOutcome,
    modelRefs: string[],
    evidenceRefs: string[],
    reasonCodes: string[],
    admissionReceiptRef: string | null = null
  ): ArbitrationReceipt {
    return {
      schemaVersion: "1.0",
      correlationId: proposal.correlationId,
      proposalHash: proposal.proposalHash,
      policyVersion: proposal.policyVersion,
      modelAssessmentRefs: modelRefs,
      evidenceRefs,
      outcome,
      reasonCodes,
      admissionReceiptRef,
    };
  }
}
