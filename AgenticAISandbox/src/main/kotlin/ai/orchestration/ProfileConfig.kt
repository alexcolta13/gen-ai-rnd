package ai.orchestration

/**
 * Specifies a concrete provider + model to use for a step or as a profile default.
 * Values must match the provider's [ai.AIProvider.availableModels] list exactly.
 */
data class ModelPreference(
    val provider: String,
    val model: String
)

/**
 * Model routing preferences scoped to a profile.
 *
 * Resolution order per step:
 *   1. stepOverrides[stepId] — iterate in order, use first available provider+model
 *   2. default              — iterate in order, use first available provider+model
 *   3. ModelRouter          — keyword-based classification + fallback chain
 *
 * Both [default] and [stepOverrides] values are ordered lists: the engine tries each
 * entry in sequence and picks the first one whose provider is available and whose
 * model is present in that provider's model list.
 */
data class ModelPreferences(
    /** Ordered preference list applied to every step that has no step-level override. */
    val default: List<ModelPreference>?,
    /** Map of step ID → ordered preference list, overrides the profile default for that step only. */
    val stepOverrides: Map<String, List<ModelPreference>>?
)

/**
 * A context key-value that the CLI should prompt the user for before running the profile.
 * The collected value is injected as {{key}} in step templates.
 */
data class ContextPrompt(
    val key: String,
    val prompt: String
)

/**
 * Configuration for a single orchestration step.
 *
 * Template syntax: use {{variable}} placeholders in [systemPrompt] and [userPromptTemplate].
 *
 * Built-in variables available in every step:
 *   {{input}}          — the user's original input
 *   {{step_<outputKey>}} — the output of a prior step (by its outputKey)
 *
 * Context variables (from contextPrompts or CLI flags) are also injected, e.g. {{from}}, {{to}}.
 */
data class StepConfig(
    val id: String,
    val name: String,
    val systemPrompt: String,
    val userPromptTemplate: String,
    /** Key under which this step's output is stored for use by later steps. Defaults to [id]. */
    val outputKey: String?
) {
    val resolvedOutputKey: String get() = outputKey ?: id
}

/**
 * Top-level profile configuration loaded from a JSON file in the profiles/ directory.
 */
data class ProfileConfig(
    val name: String,
    val description: String,
    /** Context the user is prompted to provide before the profile runs (e.g., source/target language). */
    val contextPrompts: List<ContextPrompt>?,
    val steps: List<StepConfig>,
    /** Model routing preferences for this profile. Null means fall through to ModelRouter for every step. */
    val modelPreferences: ModelPreferences?
)
