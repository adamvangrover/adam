permit(
    principal,
    action == Action::"ExecuteTask",
    resource
) when {
    principal.role == "Orchestrator" &&
    resource.classification != "Restricted_MNPI"
};
