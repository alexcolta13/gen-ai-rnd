package mcp

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import github.GitHubClient
import github.InlineComment

/**
 * MCP server exposing GitHub PR operations as tools.
 * Compatible with Claude Desktop and other MCP clients.
 */
class GitHubMcpServer : McpServer() {

    private val client = GitHubClient()

    override fun listTools(): List<McpTool> = listOf(
        McpTool(
            name        = "github_get_pr",
            description = "Fetch a pull request's metadata, description, and file diffs from GitHub.",
            inputSchema = schema(
                required = listOf("repo", "pr"),
                properties = mapOf(
                    "repo" to stringProp("Repository in 'owner/repo' format, e.g. 'octocat/Hello-World'"),
                    "pr"   to intProp("Pull request number")
                )
            )
        ),
        McpTool(
            name        = "github_post_review",
            description = "Post a formal pull request review (with optional inline comments) to GitHub.",
            inputSchema = schema(
                required = listOf("repo", "pr", "body", "event"),
                properties = mapOf(
                    "repo"     to stringProp("Repository in 'owner/repo' format"),
                    "pr"       to intProp("Pull request number"),
                    "body"     to stringProp("Overall review body text"),
                    "event"    to stringProp("Review event: COMMENT, APPROVE, or REQUEST_CHANGES"),
                    "comments" to stringProp(
                        "Optional JSON array of inline comments, e.g. [{\"path\":\"src/Foo.kt\",\"line\":10,\"body\":\"...\"}]"
                    )
                )
            )
        ),
        McpTool(
            name        = "github_post_comment",
            description = "Post a plain comment on a GitHub pull request (issue comment).",
            inputSchema = schema(
                required = listOf("repo", "pr", "body"),
                properties = mapOf(
                    "repo" to stringProp("Repository in 'owner/repo' format"),
                    "pr"   to intProp("Pull request number"),
                    "body" to stringProp("Comment body text")
                )
            )
        ),
        McpTool(
            name        = "github_get_file",
            description = "Retrieve the decoded content of a file from a GitHub repository at a specific ref.",
            inputSchema = schema(
                required = listOf("repo", "path", "ref"),
                properties = mapOf(
                    "repo" to stringProp("Repository in 'owner/repo' format"),
                    "path" to stringProp("File path within the repository, e.g. 'src/Main.kt'"),
                    "ref"  to stringProp("Branch name, tag, or commit SHA")
                )
            )
        )
    )

    override fun callTool(name: String, args: JsonObject): String {
        return when (name) {
            "github_get_pr"       -> toolGetPR(args)
            "github_post_review"  -> toolPostReview(args)
            "github_post_comment" -> toolPostComment(args)
            "github_get_file"     -> toolGetFile(args)
            else                  -> throw IllegalArgumentException("Unknown tool: $name")
        }
    }

    // -------------------------------------------------------------------------
    // Tool implementations
    // -------------------------------------------------------------------------

    private fun toolGetPR(args: JsonObject): String {
        val repo = args.get("repo")?.asString ?: throw IllegalArgumentException("Missing 'repo'")
        val pr   = args.get("pr")?.asInt     ?: throw IllegalArgumentException("Missing 'pr'")

        val prData  = client.getPR(repo, pr)
        val files   = client.getPRFiles(repo, pr)

        val sb = StringBuilder()
        sb.appendLine("# PR #${prData.number}: ${prData.title}")
        sb.appendLine("URL: ${prData.htmlUrl}")
        sb.appendLine("Base: ${prData.base.ref} ← Head: ${prData.head.ref} (${prData.head.sha.take(7)})")
        if (!prData.body.isNullOrBlank()) {
            sb.appendLine()
            sb.appendLine("## Description")
            sb.appendLine(prData.body)
        }
        sb.appendLine()
        sb.appendLine("## Files Changed (${files.size})")
        for (file in files) {
            sb.appendLine()
            sb.appendLine("### ${file.filename} [${file.status}] (+${file.additions}/-${file.deletions})")
            if (file.patch != null) {
                sb.appendLine("```diff")
                sb.appendLine(file.patch)
                sb.appendLine("```")
            }
        }
        return sb.toString()
    }

    private fun toolPostReview(args: JsonObject): String {
        val repo     = args.get("repo")?.asString  ?: throw IllegalArgumentException("Missing 'repo'")
        val prNumber = args.get("pr")?.asInt        ?: throw IllegalArgumentException("Missing 'pr'")
        val body     = args.get("body")?.asString   ?: throw IllegalArgumentException("Missing 'body'")
        val event    = args.get("event")?.asString  ?: "COMMENT"

        val prData = client.getPR(repo, prNumber)

        val comments: List<InlineComment> = args.get("comments")?.let { el ->
            if (el.isJsonNull) return@let emptyList()
            val jsonStr = if (el.isJsonArray) el.toString() else el.asString
            try {
                val arr = JsonParser.parseString(jsonStr).asJsonArray
                val gson = com.google.gson.GsonBuilder()
                    .setFieldNamingPolicy(com.google.gson.FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
                    .create()
                val type = object : com.google.gson.reflect.TypeToken<List<InlineComment>>() {}.type
                gson.fromJson(arr, type)
            } catch (e: Exception) {
                emptyList()
            }
        } ?: emptyList()

        client.postReview(repo, prNumber, prData.head.sha, body, event, comments)
        return "Review posted successfully to PR #$prNumber (${prData.htmlUrl})"
    }

    private fun toolPostComment(args: JsonObject): String {
        val repo     = args.get("repo")?.asString ?: throw IllegalArgumentException("Missing 'repo'")
        val prNumber = args.get("pr")?.asInt      ?: throw IllegalArgumentException("Missing 'pr'")
        val body     = args.get("body")?.asString ?: throw IllegalArgumentException("Missing 'body'")

        client.postIssueComment(repo, prNumber, body)
        return "Comment posted successfully to PR #$prNumber"
    }

    private fun toolGetFile(args: JsonObject): String {
        val repo = args.get("repo")?.asString ?: throw IllegalArgumentException("Missing 'repo'")
        val path = args.get("path")?.asString ?: throw IllegalArgumentException("Missing 'path'")
        val ref  = args.get("ref")?.asString  ?: throw IllegalArgumentException("Missing 'ref'")

        return client.getFileContent(repo, path, ref)
    }

    // -------------------------------------------------------------------------
    // Schema helpers
    // -------------------------------------------------------------------------

    private fun schema(required: List<String>, properties: Map<String, JsonObject>): JsonObject {
        val propsObj = JsonObject()
        for ((key, value) in properties) propsObj.add(key, value)

        val reqArray = com.google.gson.JsonArray()
        for (r in required) reqArray.add(r)

        return JsonObject().apply {
            addProperty("type", "object")
            add("properties", propsObj)
            add("required", reqArray)
        }
    }

    private fun stringProp(description: String) = JsonObject().apply {
        addProperty("type", "string")
        addProperty("description", description)
    }

    private fun intProp(description: String) = JsonObject().apply {
        addProperty("type", "integer")
        addProperty("description", description)
    }
}
