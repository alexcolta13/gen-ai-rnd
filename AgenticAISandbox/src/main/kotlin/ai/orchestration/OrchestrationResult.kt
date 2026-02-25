package ai.orchestration

/** Output of a single executed step. */
data class StepResult(
    val stepId: String,
    val stepName: String,
    val provider: String,
    val model: String,
    val output: String
)

/** Aggregated output of a complete profile run. */
data class OrchestrationResult(
    val profileName: String,
    val steps: List<StepResult>,
    /** The output of the final step — the primary result surfaced to the user. */
    val finalOutput: String
)
