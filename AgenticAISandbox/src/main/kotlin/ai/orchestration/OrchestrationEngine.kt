package ai.orchestration

import ai.AIProvider
import ai.ChatMessage
import ai.ProviderRegistry
import ai.router.ModelRouter

/**
 * Executes an [ProfileConfig] step-by-step, routing each step to the correct
 * provider and model, rendering {{variable}} templates, and accumulating step
 * outputs for use by subsequent steps.
 *
 * Model resolution order per step:
 *   1. profileConfig.modelPreferences.stepOverrides[step.id]  — first available in list
 *   2. profileConfig.modelPreferences.default                 — first available in list
 *   3. ModelRouter (classify + fallback chain)
 */
class OrchestrationEngine {
    private val router = ModelRouter()

    /**
     * Runs all steps in [config] sequentially.
     *
     * @param input          The primary user input (code, text, etc.). Available as {{input}}.
     * @param context        Extra key-value pairs (e.g., "from" / "to"). Available as {{key}}.
     * @param onStepStart    Called before each step is sent to the API.
     * @param onStepComplete Called after each step completes successfully.
     */
    fun execute(
        config: ProfileConfig,
        input: String,
        context: Map<String, String> = emptyMap(),
        onStepStart: (index: Int, total: Int, step: StepConfig) -> Unit = { _, _, _ -> },
        onStepComplete: (index: Int, total: Int, result: StepResult) -> Unit = { _, _, _ -> }
    ): OrchestrationResult {
        val steps = config.steps
        val total = steps.size

        // vars holds all substitution values: input, context keys, and step outputs as they accumulate
        val vars = mutableMapOf<String, String>()
        vars["input"] = input
        vars.putAll(context)

        val results = mutableListOf<StepResult>()

        steps.forEachIndexed { idx, step ->
            onStepStart(idx + 1, total, step)

            val systemPrompt = renderTemplate(step.systemPrompt, vars)
            val userPrompt   = renderTemplate(step.userPromptTemplate, vars)

            val messages = buildList {
                if (systemPrompt.isNotBlank()) add(ChatMessage("system", systemPrompt))
                add(ChatMessage("user", userPrompt))
            }

            val (provider, model) = resolveModel(config, step, messages)
            val response = provider.chat(messages, model)

            val result = StepResult(
                stepId   = step.id,
                stepName = step.name,
                provider = provider.name,
                model    = model,
                output   = response.content
            )
            results.add(result)

            // Make this step's output available to all subsequent steps
            vars["step_${step.resolvedOutputKey}"] = response.content

            onStepComplete(idx + 1, total, result)
        }

        return OrchestrationResult(
            profileName = config.name,
            steps       = results,
            finalOutput = results.lastOrNull()?.output ?: ""
        )
    }

    // -------------------------------------------------------------------------
    // Template rendering
    // -------------------------------------------------------------------------

    /** Replaces every {{key}} occurrence in [template] with its value from [vars]. */
    private fun renderTemplate(template: String, vars: Map<String, String>): String {
        var result = template
        for ((key, value) in vars) {
            result = result.replace("{{$key}}", value)
        }
        return result
    }

    // -------------------------------------------------------------------------
    // Model resolution
    // -------------------------------------------------------------------------

    private fun resolveModel(
        config: ProfileConfig,
        step: StepConfig,
        messages: List<ChatMessage>
    ): Pair<AIProvider, String> {
        // 1. Step-level preference list — first available wins
        val stepPrefs = config.modelPreferences?.stepOverrides?.get(step.id)
        firstAvailable(stepPrefs)?.let { return it }

        // 2. Profile-level default preference list — first available wins
        firstAvailable(config.modelPreferences?.default)?.let { return it }

        // 3. ModelRouter fallback
        val decision = router.decide(messages)
            ?: throw RuntimeException("No available provider for step '${step.name}'")
        return Pair(decision.provider, decision.model)
    }

    /**
     * Iterates [prefs] in order and returns the first entry whose provider is
     * available and whose model exists in that provider's model list, or null if
     * none qualify.
     */
    private fun firstAvailable(prefs: List<ModelPreference>?): Pair<AIProvider, String>? {
        if (prefs.isNullOrEmpty()) return null
        for (pref in prefs) {
            val provider = ProviderRegistry.getByName(pref.provider) ?: continue
            if (provider.isAvailable() && pref.model in provider.availableModels) {
                return Pair(provider, pref.model)
            }
        }
        return null
    }
}
