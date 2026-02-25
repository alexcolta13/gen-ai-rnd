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

class OllamaProvider : AIProvider {
    override val name = "ollama"

    private val baseUrl get() = Config.ollamaBaseUrl

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()
    private val gson = Gson()
    private val json = "application/json; charset=utf-8".toMediaType()

    /** The shared model catalog — static metadata about well-known free models. */
    val catalog: OllamaModelCatalog get() = OllamaModelCatalog

    /**
     * Returns installed model names. Only installed models count as "available"
     * for routing (router checks model presence in this list).
     */
    override val availableModels: List<String>
        get() = fetchInstalledModels()

    private fun fetchInstalledModels(): List<String> {
        return try {
            val request = Request.Builder()
                .url("$baseUrl/api/tags")
                .get()
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return emptyList()
                val body = response.body?.string() ?: return emptyList()
                val json = gson.fromJson(body, JsonObject::class.java)
                json.getAsJsonArray("models")
                    ?.mapNotNull { it.asJsonObject.get("name")?.asString }
                    ?: emptyList()
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    override fun isAvailable(): Boolean {
        return try {
            val request = Request.Builder()
                .url(baseUrl)
                .get()
                .build()
            client.newCall(request).execute().use { response ->
                response.isSuccessful || response.code in 400..499
            }
        } catch (e: Exception) {
            false
        }
    }

    override fun chat(messages: List<ChatMessage>, model: String): ChatResponse {
        val body = gson.toJson(mapOf(
            "model" to model,
            "messages" to messages.map { mapOf("role" to it.role, "content" to it.content) },
            "stream" to false
        ))

        val request = Request.Builder()
            .url("$baseUrl/api/chat")
            .post(body.toRequestBody(json))
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: ""
                throw RuntimeException("Ollama API error ${response.code}: $errorBody")
            }
            val responseBody = response.body?.string()
                ?: throw RuntimeException("Empty response from Ollama")
            val jsonResponse = gson.fromJson(responseBody, JsonObject::class.java)
            val content = jsonResponse
                .getAsJsonObject("message")
                .get("content").asString
            return ChatResponse(content = content, model = model, provider = name)
        }
    }
}
