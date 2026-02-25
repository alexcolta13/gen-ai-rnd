package config

import io.github.cdimascio.dotenv.dotenv

object Config {
    private val dotenv = runCatching {
        dotenv {
            ignoreIfMissing = true
            ignoreIfMalformed = true
        }
    }.getOrNull()

    private fun get(key: String): String? =
        System.getenv(key) ?: dotenv?.get(key)

    val openAiApiKey: String? get() = get("OPENAI_API_KEY")
    val anthropicApiKey: String? get() = get("ANTHROPIC_API_KEY")
    val ollamaBaseUrl: String get() = get("OLLAMA_BASE_URL") ?: "http://localhost:11434"
    val githubToken: String? get() = get("GITHUB_TOKEN")
}
