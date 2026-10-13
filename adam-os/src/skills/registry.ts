export interface SkillManifest {
  schema_version: "1.0";
  skill_id: string;
  version: string;
  entrypoint: string;
  input_schema: string;
  output_schema: string;
  execution: {
    timeout_seconds: number;
    max_attempts: number;
    network_access: boolean;
    filesystem_scope: "read_only" | "repository_read_only" | "ephemeral_write" | "full";
  };
  capabilities: string[];
  side_effect_class: "read_only" | "append_only_audit" | "state_mutation";
  authority_level: "proposal_only" | "diagnostic_only" | "admitted_authority";
}

export class SkillRegistry {
  private skills: Map<string, SkillManifest> = new Map();

  public registerSkill(manifest: SkillManifest): void {
    if (!manifest.skill_id || !manifest.entrypoint) {
      throw new Error("Invalid skill manifest: missing skill_id or entrypoint");
    }
    this.skills.set(manifest.skill_id, manifest);
  }

  public getSkill(skillId: string): SkillManifest | undefined {
    return this.skills.get(skillId);
  }

  public listSkills(): SkillManifest[] {
    return Array.from(this.skills.values());
  }

  public validateCapabilities(skillId: string, requiredCapabilities: string[]): boolean {
    const skill = this.getSkill(skillId);
    if (!skill) return false;
    return requiredCapabilities.every((cap) => skill.capabilities.includes(cap));
  }
}
