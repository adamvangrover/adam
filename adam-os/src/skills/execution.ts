import { SkillManifest } from "./registry";

export interface SkillExecutionContext {
  skill_id: string;
  skill_version: string;
  run_id: string;
  correlation_id: string;
  idempotency_key: string;
  input_data: Record<string, unknown>;
  policy_version: string;
  execution_scope: string;
  provided_capabilities: string[];
}

export interface SkillExecutionResultEnvelope<T = Record<string, unknown>> {
  skill_id: string;
  skill_version: string;
  run_id: string;
  correlation_id: string;
  idempotency_key: string;
  input_content_hash: string;
  output_content_hash: string;
  precondition_passed: boolean;
  postcondition_passed: boolean;
  capability_decision: "GRANTED" | "DENIED";
  provenance_record_ids: string[];
  status: "COMPLETED" | "REPLAYED" | "REJECTED" | "QUARANTINED" | "NON_AUTHORITATIVE_RESULT";
  result_data: T | null;
  error_code?: string;
  error_message?: string;
}

export class SkillExecutionRunner {
  private idempotencyLedger: Map<string, SkillExecutionResultEnvelope> = new Map();

  public generateIdempotencyKey(
    skillId: string,
    skillVersion: string,
    inputHash: string,
    policyVersion: string,
    scope: string
  ): string {
    return `${skillId}:${skillVersion}:${inputHash}:${policyVersion}:${scope}`;
  }

  public executeSkill<T = Record<string, unknown>>(
    manifest: SkillManifest,
    context: SkillExecutionContext,
    inputHash: string,
    hashCalculator: (obj: unknown) => string,
    executor: (ctx: SkillExecutionContext) => { data: T; provenance: string[] }
  ): SkillExecutionResultEnvelope<T> {
    const key = context.idempotency_key;

    // Idempotency check: replay prior result if already executed
    if (this.idempotencyLedger.has(key)) {
      const existing = this.idempotencyLedger.get(key) as SkillExecutionResultEnvelope<T>;
      return {
        ...existing,
        status: "REPLAYED",
      };
    }

    // Capability check: caller must possess all required capabilities declared in manifest
    const missingCapabilities = manifest.capabilities.filter(
      (cap) => !context.provided_capabilities.includes(cap)
    );
    if (missingCapabilities.length > 0) {
      return {
        skill_id: manifest.skill_id,
        skill_version: manifest.version,
        run_id: context.run_id,
        correlation_id: context.correlation_id,
        idempotency_key: key,
        input_content_hash: inputHash,
        output_content_hash: "",
        precondition_passed: true,
        postcondition_passed: false,
        capability_decision: "DENIED",
        provenance_record_ids: [],
        status: "REJECTED",
        result_data: null,
        error_code: "CAPABILITY_DENIED",
        error_message: `Skill execution denied missing capabilities: ${missingCapabilities.join(", ")}`,
      };
    }

    try {
      const executionOutcome = executor(context);
      const outputHash = hashCalculator(executionOutcome.data);

      const envelope: SkillExecutionResultEnvelope<T> = {
        skill_id: manifest.skill_id,
        skill_version: manifest.version,
        run_id: context.run_id,
        correlation_id: context.correlation_id,
        idempotency_key: key,
        input_content_hash: inputHash,
        output_content_hash: outputHash,
        precondition_passed: true,
        postcondition_passed: true,
        capability_decision: "GRANTED",
        provenance_record_ids: executionOutcome.provenance,
        status: "COMPLETED",
        result_data: executionOutcome.data,
      };

      this.idempotencyLedger.set(key, envelope as unknown as SkillExecutionResultEnvelope);
      return envelope;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      return {
        skill_id: manifest.skill_id,
        skill_version: manifest.version,
        run_id: context.run_id,
        correlation_id: context.correlation_id,
        idempotency_key: key,
        input_content_hash: inputHash,
        output_content_hash: "",
        precondition_passed: true,
        postcondition_passed: false,
        capability_decision: "GRANTED",
        provenance_record_ids: [],
        status: "QUARANTINED",
        result_data: null,
        error_code: "EXECUTION_FAILURE",
        error_message: errorMessage,
      };
    }
  }
}
