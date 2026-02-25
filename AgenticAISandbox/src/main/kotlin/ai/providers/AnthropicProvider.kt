package ai.providers

import ai.AIProvider
import ai.ChatMessage
import ai.ChatResponse
import com.google.gson.Gson
import com.google.gson.JsonObject
import config.Config
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

class AnthropicProvider : AIProvider {
    override val name = "anthropic"
    override val availableModels = listOf("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-3-5")

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()
    private val gson = Gson()
    private val json = "application/json; charset=utf-8".toMediaType()

    override fun isAvailable(): Boolean = !Config.anthropicApiKey.isNullOrBlank()

    override fun chat(messages: List<ChatMessage>, model: String): ChatResponse {
        val apiKey = Config.anthropicApiKey
            ?: throw IllegalStateException("ANTHROPIC_API_KEY is not set")

        // Separate system messages from user/assistant turns
        val systemContent = messages.filter { it.role == "system" }.joinToString("\n") { it.content }
        val conversationMessages = messages.filter { it.role != "system" }
            .map { mapOf("role" to it.role, "content" to it.content) }

        val bodyMap: MutableMap<String, Any> = mutableMapOf(
            "model" to model,
            "max_tokens" to 4096,
            "messages" to conversationMessages
        )
        if (systemContent.isNotBlank()) {
            bodyMap["system"] = systemContent
        }

        val body = gson.toJson(bodyMap)

        val request = Request.Builder()
            .url("https://api.anthropic.com/v1/messages")
            .addHeader("x-api-key", apiKey)
            .addHeader("anthropic-version", "2023-06-01")
            .post(body.toRequestBody(json))
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: ""
                throw RuntimeException("Anthropic API error ${response.code}: $errorBody")
            }
            val responseBody = response.body?.string()
                ?: throw RuntimeException("Empty response from Anthropic")
            val jsonResponse = gson.fromJson(responseBody, JsonObject::class.java)
            val content = jsonResponse
                .getAsJsonArray("content")
                .get(0).asJsonObject
                .get("text").asString
            return ChatResponse(content = content, model = model, provider = name)
        }
    }
}
