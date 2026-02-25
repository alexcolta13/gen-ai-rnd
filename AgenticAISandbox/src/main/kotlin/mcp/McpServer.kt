package mcp

import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser

data class McpTool(
    val name: String,
    val description: String,
    val inputSchema: JsonObject
)

/**
 * Base JSON-RPC 2.0 stdio server implementing the Model Context Protocol (MCP).
 * Subclasses implement [listTools] and [callTool] to expose domain-specific tools.
 *
 * Protocol version: 2024-11-05
 */
abstract class McpServer {

    private val gson = GsonBuilder().disableHtmlEscaping().create()

    abstract fun listTools(): List<McpTool>

    /**
     * Called when the client invokes a tool.
     * @param name the tool name
     * @param args the tool arguments as a JsonObject (may be empty)
     * @return text content to return in the response
     */
    abstract fun callTool(name: String, args: JsonObject): String

    /** Starts the stdio read-loop. Blocks until stdin is closed. */
    fun run() {
        val reader = System.`in`.bufferedReader()
        var line: String?
        while (reader.readLine().also { line = it } != null) {
            val trimmed = line!!.trim()
            if (trimmed.isEmpty()) continue
            val response = handleMessage(trimmed)
            if (response != null) {
                println(response)
                System.out.flush()
            }
        }
    }

    // -------------------------------------------------------------------------
    // Message dispatch
    // -------------------------------------------------------------------------

    private fun handleMessage(raw: String): String? {
        return try {
            val request = JsonParser.parseString(raw).asJsonObject
            val id      = request.get("id")
            val method  = request.get("method")?.asString ?: return errorResponse(null, -32600, "Missing method")

            when (method) {
                "initialize"              -> handleInitialize(id, request)
                "notifications/initialized" -> null  // notification — no response
                "tools/list"              -> handleToolsList(id)
                "tools/call"              -> handleToolsCall(id, request)
                else                      -> errorResponse(id, -32601, "Method not found: $method")
            }
        } catch (e: Exception) {
            errorResponse(null, -32700, "Parse error: ${e.message}")
        }
    }

    private fun handleInitialize(id: com.google.gson.JsonElement?, request: JsonObject): String {
        val result = JsonObject().apply {
            addProperty("protocolVersion", "2024-11-05")
            add("capabilities", JsonObject().apply {
                add("tools", JsonObject())
            })
            add("serverInfo", JsonObject().apply {
                addProperty("name", "github-mcp-server")
                addProperty("version", "1.0.0")
            })
        }
        return successResponse(id, result)
    }

    private fun handleToolsList(id: com.google.gson.JsonElement?): String {
        val toolsArray = com.google.gson.JsonArray()
        for (tool in listTools()) {
            toolsArray.add(JsonObject().apply {
                addProperty("name", tool.name)
                addProperty("description", tool.description)
                add("inputSchema", tool.inputSchema)
            })
        }
        val result = JsonObject().apply {
            add("tools", toolsArray)
        }
        return successResponse(id, result)
    }

    private fun handleToolsCall(id: com.google.gson.JsonElement?, request: JsonObject): String {
        val params = request.getAsJsonObject("params") ?: JsonObject()
        val name   = params.get("name")?.asString
            ?: return errorResponse(id, -32602, "Missing tool name")
        val args   = params.getAsJsonObject("arguments") ?: JsonObject()

        return try {
            val text = callTool(name, args)
            val contentArray = com.google.gson.JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("type", "text")
                    addProperty("text", text)
                })
            }
            val result = JsonObject().apply {
                add("content", contentArray)
                addProperty("isError", false)
            }
            successResponse(id, result)
        } catch (e: Exception) {
            val contentArray = com.google.gson.JsonArray().apply {
                add(JsonObject().apply {
                    addProperty("type", "text")
                    addProperty("text", "Error: ${e.message}")
                })
            }
            val result = JsonObject().apply {
                add("content", contentArray)
                addProperty("isError", true)
            }
            successResponse(id, result)
        }
    }

    // -------------------------------------------------------------------------
    // Response builders
    // -------------------------------------------------------------------------

    private fun successResponse(id: com.google.gson.JsonElement?, result: JsonObject): String {
        val response = JsonObject().apply {
            addProperty("jsonrpc", "2.0")
            if (id != null) add("id", id) else addProperty("id", null as String?)
            add("result", result)
        }
        return gson.toJson(response)
    }

    private fun errorResponse(id: com.google.gson.JsonElement?, code: Int, message: String): String {
        val response = JsonObject().apply {
            addProperty("jsonrpc", "2.0")
            if (id != null) add("id", id) else addProperty("id", null as String?)
            add("error", JsonObject().apply {
                addProperty("code", code)
                addProperty("message", message)
            })
        }
        return gson.toJson(response)
    }
}
