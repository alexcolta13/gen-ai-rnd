package ai

data class ChatMessage(val role: String, val content: String)

data class ChatResponse(val content: String, val model: String, val provider: String)

interface AIProvider {
    val name: String
    val availableModels: List<String>
    fun chat(messages: List<ChatMessage>, model: String): ChatResponse
    fun isAvailable(): Boolean
}
