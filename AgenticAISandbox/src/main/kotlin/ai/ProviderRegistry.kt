package ai

import ai.providers.AnthropicProvider
import ai.providers.OllamaProvider
import ai.providers.OpenAIProvider

object ProviderRegistry {
    private val all: List<AIProvider> = listOf(
        OpenAIProvider(),
        AnthropicProvider(),
        OllamaProvider()
    )

    /** All registered providers, regardless of availability. */
    fun allProviders(): List<AIProvider> = all

    fun availableProviders(): List<AIProvider> = all.filter { it.isAvailable() }

    fun getByName(name: String): AIProvider? =
        all.find { it.name.equals(name, ignoreCase = true) }
}
