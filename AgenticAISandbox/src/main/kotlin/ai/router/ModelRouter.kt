package ai.router

import ai.AIProvider
import ai.ChatMessage
import ai.ChatResponse
import ai.ProviderRegistry

/**
 * Routes each chat request to the most appropriate provider and model based on
 * the content of the messages. Falls back through an ordered list of candidates
 * until one succeeds.
 *
 * Cloud providers are tried first; Ollama fallbacks use specific preferred models
 * (from OllamaModelCatalog) before falling back to any installed Ollama model.
 *
 * Routing table (first available wins):
 *   CODING   → anthropic/claude-sonnet-4-6 → openai/gpt-4o
 *              → ollama/qwen2.5-coder:7b → ollama/phi4:14b → ollama/codellama:7b → ollama (any)
 *   ANALYSIS → anthropic/claude-opus-4-6   → openai/gpt-4o → anthropic/claude-sonnet-4-6
 *              → ollama/deepseek-r1:7b → ollama/phi4:14b → ollama/llama3.1:8b → ollama (any)
 *   CREATIVE → openai/gpt-4o               → anthropic/claude-sonnet-4-6
 *              → ollama/llama3.1:8b → ollama/mistral:7b → ollama (any)
 *   SIMPLE   → openai/gpt-4o-mini          → anthropic/claude-haiku-3-5
 *              → ollama/llama3.2:3b → ollama/phi4-mini:3.8b → ollama (any)
 *   GENERAL  → anthropic/claude-sonnet-4-6 → openai/gpt-4o
 *              → ollama/llama3.1:8b → ollama/llama3.2:3b → ollama (any)
 */
class ModelRouter {

    private val routingTable: Map<TaskType, List<RouteTarget>> = mapOf(
        TaskType.CODING to listOf(
            RouteTarget("anthropic", "claude-sonnet-4-6"),
            RouteTarget("openai",    "gpt-4o"),
            RouteTarget("ollama",    "qwen2.5-coder:7b"),
            RouteTarget("ollama",    "phi4:14b"),
            RouteTarget("ollama",    "qwen2.5-coder:14b"),
            RouteTarget("ollama",    "codellama:7b"),
            RouteTarget("ollama",    "deepseek-r1:7b"),
            RouteTarget("ollama",    "llama3.1:8b"),
            RouteTarget("ollama")
        ),
        TaskType.ANALYSIS to listOf(
            RouteTarget("anthropic", "claude-opus-4-6"),
            RouteTarget("openai",    "gpt-4o"),
            RouteTarget("anthropic", "claude-sonnet-4-6"),
            RouteTarget("ollama",    "deepseek-r1:7b"),
            RouteTarget("ollama",    "phi4:14b"),
            RouteTarget("ollama",    "deepseek-r1:14b"),
            RouteTarget("ollama",    "gemma2:9b"),
            RouteTarget("ollama",    "llama3.1:8b"),
            RouteTarget("ollama")
        ),
        TaskType.CREATIVE to listOf(
            RouteTarget("openai",    "gpt-4o"),
            RouteTarget("anthropic", "claude-sonnet-4-6"),
            RouteTarget("ollama",    "llama3.1:8b"),
            RouteTarget("ollama",    "mistral:7b"),
            RouteTarget("ollama",    "mistral-small:22b"),
            RouteTarget("ollama",    "llama3.2:3b"),
            RouteTarget("ollama")
        ),
        TaskType.SIMPLE to listOf(
            RouteTarget("openai",    "gpt-4o-mini"),
            RouteTarget("anthropic", "claude-haiku-3-5"),
            RouteTarget("ollama",    "llama3.2:3b"),
            RouteTarget("ollama",    "phi4-mini:3.8b"),
            RouteTarget("ollama",    "qwen2.5:3b"),
            RouteTarget("ollama",    "llama3.1:8b"),
            RouteTarget("ollama")
        ),
        TaskType.GENERAL to listOf(
            RouteTarget("anthropic", "claude-sonnet-4-6"),
            RouteTarget("openai",    "gpt-4o"),
            RouteTarget("ollama",    "llama3.1:8b"),
            RouteTarget("ollama",    "phi4:14b"),
            RouteTarget("ollama",    "llama3.2:3b"),
            RouteTarget("ollama",    "mistral:7b"),
            RouteTarget("ollama")
        )
    )

    // -------------------------------------------------------------------------
    // Classification
    // -------------------------------------------------------------------------

    /** Classifies the conversation's user turns into a [TaskType]. */
    fun classify(messages: List<ChatMessage>): TaskType {
        val text = messages
            .filter { it.role == "user" }
            .joinToString(" ") { it.content }
            .lowercase()

        return when {
            isCoding(text)   -> TaskType.CODING
            isAnalysis(text) -> TaskType.ANALYSIS
            isCreative(text) -> TaskType.CREATIVE
            isSimple(text)   -> TaskType.SIMPLE
            else             -> TaskType.GENERAL
        }
    }

    private fun isCoding(text: String): Boolean {
        val keywords = listOf(
            "code", "function", "class", "method", "debug", "error", "bug", "fix",
            "implement", "refactor", "unit test", "algorithm", "script",
            "compile", "syntax", "variable", "recursion", "api", "database",
            "sql", "json", "html", "css", "regex", "parse", "exception",
            "import", "library", "package", "object", "interface"
        )
        return keywords.count { text.contains(it) } >= 2
    }

    private fun isAnalysis(text: String): Boolean {
        val keywords = listOf(
            "analyze", "analyse", "explain", "compare", "review", "evaluate",
            "assess", "summarize", "summarise", "pros and cons",
            "difference between", "implications", "impact"
        )
        return keywords.any { text.contains(it) } || text.split("\\s+".toRegex()).size > 80
    }

    private fun isCreative(text: String): Boolean {
        val keywords = listOf(
            "write a story", "short story", "poem", "creative writing", "imagine",
            "fictional", "narrative", "write me a", "once upon"
        )
        return keywords.any { text.contains(it) }
    }

    private fun isSimple(text: String): Boolean {
        val wordCount = text.trim().split("\\s+".toRegex()).size
        val simpleStarters = listOf(
            "what is", "who is", "define ", "translate", "convert",
            "when was", "how many", "capital of", "what's"
        )
        return wordCount <= 15 || simpleStarters.any { text.startsWith(it) }
    }

    // -------------------------------------------------------------------------
    // Routing
    // -------------------------------------------------------------------------

    /** The result of a routing decision. */
    data class RouterDecision(val taskType: TaskType, val provider: AIProvider, val model: String)

    /**
     * Determines the best [RouterDecision] for [messages] without making an API call.
     * Returns null only if no providers are available at all.
     */
    fun decide(messages: List<ChatMessage>): RouterDecision? {
        val taskType = classify(messages)
        for (target in routingTable[taskType] ?: emptyList()) {
            val (provider, model) = resolveTarget(target) ?: continue
            return RouterDecision(taskType, provider, model)
        }
        // Last-resort: first available provider with any model
        val provider = ProviderRegistry.availableProviders().firstOrNull() ?: return null
        val model = provider.availableModels.firstOrNull() ?: return null
        return RouterDecision(TaskType.GENERAL, provider, model)
    }

    /**
     * Routes [messages] to the best available provider+model, tries to chat,
     * and falls back to subsequent candidates on failure.
     *
     * @return the [ChatResponse] paired with the [RouterDecision] that produced it.
     * @throws RuntimeException if every candidate fails.
     */
    fun routeAndChat(messages: List<ChatMessage>): Pair<ChatResponse, RouterDecision> {
        val taskType = classify(messages)
        val targets = routingTable[taskType] ?: emptyList()
        var lastError: Exception? = null

        for (target in targets) {
            val resolved = resolveTarget(target) ?: continue
            val (provider, model) = resolved
            return try {
                val response = provider.chat(messages, model)
                Pair(response, RouterDecision(taskType, provider, model))
            } catch (e: Exception) {
                lastError = e
                continue
            }
        }

        // Last-resort fallback
        val provider = ProviderRegistry.availableProviders().firstOrNull()
            ?: throw lastError ?: RuntimeException("No available provider")
        val model = provider.availableModels.firstOrNull()
            ?: throw RuntimeException("No models available for ${provider.name}")
        val response = provider.chat(messages, model)
        return Pair(response, RouterDecision(taskType, provider, model))
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun resolveTarget(target: RouteTarget): Pair<AIProvider, String>? {
        val provider = ProviderRegistry.getByName(target.providerName) ?: return null
        if (!provider.isAvailable()) return null
        val model = when {
            target.model.isNotEmpty() -> {
                if (target.model in provider.availableModels) target.model else return null
            }
            else -> provider.availableModels.firstOrNull() ?: return null
        }
        return Pair(provider, model)
    }
}
