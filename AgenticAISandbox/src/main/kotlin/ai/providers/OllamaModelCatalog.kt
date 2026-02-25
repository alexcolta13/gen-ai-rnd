package ai.providers

/**
 * Metadata for a well-known free Ollama model that runs locally on
 * consumer hardware (NVIDIA CUDA or Apple Silicon / Metal).
 *
 * [minVramGb] is a practical minimum for Q4-quantised inference at reasonable speed.
 * CPU-only machines can still run any model — just slower.
 */
data class OllamaModelInfo(
    /** Exact tag passed to `ollama pull` / `ollama run`. */
    val id: String,
    val displayName: String,
    val description: String,
    /** Approximate parameter count in billions. */
    val paramsBillions: Double,
    /** Minimum VRAM (GB) for comfortable GPU-accelerated inference (Q4 quant). */
    val minVramGb: Int,
    /**
     * Primary task strengths.
     * Values mirror [ai.router.TaskType] names in lowercase: coding, analysis, creative, simple, general.
     * "reasoning" is used for chain-of-thought / thinking models.
     */
    val taskTypes: List<String>,
    /** Included in the default suggested set shown to new users. */
    val isDefault: Boolean = false
)

object OllamaModelCatalog {

    val models: List<OllamaModelInfo> = listOf(

        // ── Tier 1: Small ≤4 GB — any GPU, fast CPU ────────────────────────────
        OllamaModelInfo(
            id              = "llama3.2:3b",
            displayName     = "Llama 3.2 3B",
            description     = "Meta's compact model. Fast responses, solid general chat and simple tasks.",
            paramsBillions  = 3.0,
            minVramGb       = 2,
            taskTypes       = listOf("general", "simple"),
            isDefault       = true
        ),
        OllamaModelInfo(
            id              = "phi4-mini:3.8b",
            displayName     = "Phi-4 Mini 3.8B",
            description     = "Microsoft's reasoning-tuned small model. Punches well above its weight class.",
            paramsBillions  = 3.8,
            minVramGb       = 3,
            taskTypes       = listOf("reasoning", "general", "simple")
        ),
        OllamaModelInfo(
            id              = "qwen2.5:3b",
            displayName     = "Qwen 2.5 3B",
            description     = "Alibaba's compact multilingual model. Good for non-English and general tasks.",
            paramsBillions  = 3.0,
            minVramGb       = 2,
            taskTypes       = listOf("general", "simple")
        ),

        // ── Tier 2: Medium ≤8 GB — RTX 3060 12GB, RTX 3070 8GB, M1/M2 16GB ───
        OllamaModelInfo(
            id              = "llama3.1:8b",
            displayName     = "Llama 3.1 8B",
            description     = "Meta's flagship 8B model. Excellent all-rounder for chat, coding, and analysis.",
            paramsBillions  = 8.0,
            minVramGb       = 5,
            taskTypes       = listOf("general", "coding", "analysis"),
            isDefault       = true
        ),
        OllamaModelInfo(
            id              = "qwen2.5-coder:7b",
            displayName     = "Qwen 2.5 Coder 7B",
            description     = "Best open coding model at 7B. Trained on 5.5T tokens, supports 92 languages.",
            paramsBillions  = 7.0,
            minVramGb       = 5,
            taskTypes       = listOf("coding"),
            isDefault       = true
        ),
        OllamaModelInfo(
            id              = "mistral:7b",
            displayName     = "Mistral 7B",
            description     = "Fast, efficient model from Mistral AI. Great for chat, summarisation, and creative tasks.",
            paramsBillions  = 7.0,
            minVramGb       = 5,
            taskTypes       = listOf("general", "creative")
        ),
        OllamaModelInfo(
            id              = "deepseek-r1:7b",
            displayName     = "DeepSeek R1 7B",
            description     = "Reasoning-focused model with visible chain-of-thought. Strong for logic and analysis.",
            paramsBillions  = 7.0,
            minVramGb       = 5,
            taskTypes       = listOf("analysis", "reasoning", "coding")
        ),
        OllamaModelInfo(
            id              = "gemma2:9b",
            displayName     = "Gemma 2 9B",
            description     = "Google's Gemma 2 model. Strong general performance, clean and well-formatted outputs.",
            paramsBillions  = 9.0,
            minVramGb       = 6,
            taskTypes       = listOf("general", "analysis")
        ),
        OllamaModelInfo(
            id              = "codellama:7b",
            displayName     = "Code Llama 7B",
            description     = "Meta's code-specialised model, fine-tuned from Llama 2 on code corpora.",
            paramsBillions  = 7.0,
            minVramGb       = 5,
            taskTypes       = listOf("coding")
        ),

        // ── Tier 3: Large ≤12 GB — RTX 3060 12GB, RTX 4070, M2/M3 Pro 16GB+ ──
        OllamaModelInfo(
            id              = "phi4:14b",
            displayName     = "Phi-4 14B",
            description     = "Microsoft's flagship compact model. Exceptional reasoning and coding well beyond its size.",
            paramsBillions  = 14.0,
            minVramGb       = 9,
            taskTypes       = listOf("coding", "analysis", "reasoning", "general"),
            isDefault       = true
        ),
        OllamaModelInfo(
            id              = "qwen2.5-coder:14b",
            displayName     = "Qwen 2.5 Coder 14B",
            description     = "Premium open coding model. Top-tier scores on HumanEval and code benchmarks.",
            paramsBillions  = 14.0,
            minVramGb       = 9,
            taskTypes       = listOf("coding")
        ),
        OllamaModelInfo(
            id              = "deepseek-r1:14b",
            displayName     = "DeepSeek R1 14B",
            description     = "Larger DeepSeek reasoning model. Competes with GPT-4 class on logic and math tasks.",
            paramsBillions  = 14.0,
            minVramGb       = 9,
            taskTypes       = listOf("analysis", "reasoning", "coding")
        ),

        // ── Tier 4: XL ≤24 GB — RTX 3090, RTX 4090, M2/M3 Max/Ultra ──────────
        OllamaModelInfo(
            id              = "mistral-small:22b",
            displayName     = "Mistral Small 22B",
            description     = "Efficient large model from Mistral AI. Strong instruction following and nuanced reasoning.",
            paramsBillions  = 22.0,
            minVramGb       = 14,
            taskTypes       = listOf("general", "analysis", "creative")
        ),
        OllamaModelInfo(
            id              = "qwen2.5:32b",
            displayName     = "Qwen 2.5 32B",
            description     = "Alibaba's large multilingual model. Near-GPT-4 quality for general and analytical tasks.",
            paramsBillions  = 32.0,
            minVramGb       = 20,
            taskTypes       = listOf("general", "analysis", "creative")
        )
    )

    /** The 4 recommended defaults shown to users who have no models installed. */
    val defaults: List<OllamaModelInfo> = models.filter { it.isDefault }

    /** Ordered list of preferred model IDs for a task type (for the router fallback chain). */
    fun preferredIdsForTask(taskType: String): List<String> = when (taskType) {
        "coding"   -> listOf("qwen2.5-coder:7b", "phi4:14b", "qwen2.5-coder:14b", "codellama:7b", "deepseek-r1:7b", "llama3.1:8b")
        "analysis" -> listOf("deepseek-r1:7b", "phi4:14b", "deepseek-r1:14b", "gemma2:9b", "llama3.1:8b")
        "creative" -> listOf("llama3.1:8b", "mistral:7b", "mistral-small:22b", "llama3.2:3b")
        "simple"   -> listOf("llama3.2:3b", "phi4-mini:3.8b", "qwen2.5:3b", "llama3.1:8b")
        "reasoning"-> listOf("deepseek-r1:14b", "deepseek-r1:7b", "phi4:14b", "phi4-mini:3.8b")
        else       -> listOf("llama3.1:8b", "phi4:14b", "llama3.2:3b", "mistral:7b")  // general
    }
}
