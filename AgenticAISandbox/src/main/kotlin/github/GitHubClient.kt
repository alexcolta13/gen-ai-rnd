package github

import com.google.gson.FieldNamingPolicy
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import config.Config
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

class GitHubClient {
    private val token: String = Config.githubToken
        ?: throw IllegalStateException("GITHUB_TOKEN is not set. Add it to your .env or environment.")

    private val http = OkHttpClient()
    private val gson = GsonBuilder()
        .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
        .create()

    private val jsonMedia = "application/json; charset=utf-8".toMediaType()
    private val baseUrl = "https://api.github.com"

    // -------------------------------------------------------------------------
    // GET helpers
    // -------------------------------------------------------------------------

    private fun get(path: String): String {
        val request = Request.Builder()
            .url("$baseUrl$path")
            .header("Authorization", "Bearer $token")
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .get()
            .build()
        return http.newCall(request).execute().use { response ->
            val body = response.body?.string() ?: ""
            if (!response.isSuccessful) {
                throw RuntimeException("GitHub GET $path failed (${response.code}): $body")
            }
            body
        }
    }

    private fun post(path: String, jsonBody: String): String {
        val body = jsonBody.toRequestBody(jsonMedia)
        val request = Request.Builder()
            .url("$baseUrl$path")
            .header("Authorization", "Bearer $token")
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .post(body)
            .build()
        return http.newCall(request).execute().use { response ->
            val responseBody = response.body?.string() ?: ""
            if (!response.isSuccessful) {
                throw RuntimeException("GitHub POST $path failed (${response.code}): $responseBody")
            }
            responseBody
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    fun getPR(repo: String, prNumber: Int): PullRequest {
        val (owner, name) = splitRepo(repo)
        val json = get("/repos/$owner/$name/pulls/$prNumber")
        return gson.fromJson(json, PullRequest::class.java)
    }

    fun getPRFiles(repo: String, prNumber: Int): List<PullRequestFile> {
        val (owner, name) = splitRepo(repo)
        val json = get("/repos/$owner/$name/pulls/$prNumber/files")
        val type = object : TypeToken<List<PullRequestFile>>() {}.type
        return gson.fromJson(json, type)
    }

    fun getFileContent(repo: String, path: String, ref: String): String {
        val (owner, name) = splitRepo(repo)
        val json = get("/repos/$owner/$name/contents/$path?ref=$ref")
        // GitHub returns base64-encoded content
        val obj = gson.fromJson(json, Map::class.java)
        val encoded = (obj["content"] as? String)
            ?: throw RuntimeException("No content field in response for $path")
        return String(java.util.Base64.getMimeDecoder().decode(encoded))
    }

    /**
     * Posts a formal PR review with optional inline comments.
     * [event] should be one of: COMMENT, APPROVE, REQUEST_CHANGES.
     */
    fun postReview(
        repo: String,
        prNumber: Int,
        commitSha: String,
        body: String,
        event: String = "COMMENT",
        comments: List<InlineComment> = emptyList()
    ) {
        val (owner, name) = splitRepo(repo)
        val commentsArray = comments.map { c ->
            mapOf("path" to c.path, "line" to c.line, "body" to c.body, "side" to c.side)
        }
        val payload = mapOf(
            "commit_id" to commitSha,
            "body"       to body,
            "event"      to event,
            "comments"   to commentsArray
        )
        post("/repos/$owner/$name/pulls/$prNumber/reviews", gson.toJson(payload))
    }

    /** Posts a plain issue comment (fallback when review API fails). */
    fun postIssueComment(repo: String, prNumber: Int, body: String) {
        val (owner, name) = splitRepo(repo)
        val payload = mapOf("body" to body)
        post("/repos/$owner/$name/issues/$prNumber/comments", gson.toJson(payload))
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private fun splitRepo(repo: String): Pair<String, String> {
        val parts = repo.split("/")
        require(parts.size == 2) { "repo must be in 'owner/name' format, got: $repo" }
        return parts[0] to parts[1]
    }
}
