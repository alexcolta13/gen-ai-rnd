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

class OpenAIProvider : AIProvider {
    override val name = "openai"
    override val availableModels = listOf("gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .build()
    private val gson = Gson()
    private val json = "application/json; charset=utf-8".toMediaType()

    override fun isAvailable(): Boolean = !Config.openAiApiKey.isNullOrBlank()

    override fun chat(messages: List<ChatMessage>, model: String): ChatResponse {
        val apiKey = Config.openAiApiKey
            ?: throw IllegalStateException("OPENAI_API_KEY is not set")

        val body = gson.toJson(mapOf(
            "model" to model,
            "messages" to messages.map { mapOf("role" to it.role, "content" to it.content) }
        ))

        val request = Request.Builder()
            .url("https://api.openai.com/v1/chat/completions")
            .addHeader("Authorization", "Bearer $apiKey")
            .post(body.toRequestBody(json))
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: ""
                throw RuntimeException("OpenAI API error ${response.code}: $errorBody")
            }
            val responseBody = response.body?.string()
                ?: throw RuntimeException("Empty response from OpenAI")
            val jsonResponse = gson.fromJson(responseBody, JsonObject::class.java)
            val content = jsonResponse
                .getAsJsonArray("choices")
                .get(0).asJsonObject
                .getAsJsonObject("message")
                .get("content").asString
            return ChatResponse(content = content, model = model, provider = name)
        }
    }
}
