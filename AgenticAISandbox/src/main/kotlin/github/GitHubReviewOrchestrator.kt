package github

import ai.orchestration.OrchestrationEngine
import ai.orchestration.ProfileLoader
import ai.orchestration.StepConfig

class GitHubReviewOrchestrator {
    private val client = GitHubClient()
    private val engine = OrchestrationEngine()

    // -------------------------------------------------------------------------
    // PR Review
    // -------------------------------------------------------------------------

    /**
     * Fetches PR data from GitHub, runs the `github-pr-review` orchestration
     * profile, parses inline comments from the final output, and posts the
     * review back to GitHub.
     *
     * @param repo          "owner/repo" format
     * @param prNumber      Pull request number
     * @param onStepUpdate  Called before each orchestration step with a status message
     */
    fun reviewPR(
        repo: String,
        prNumber: Int,
        onStepUpdate: (String) -> Unit = {}
    ): String {
        onStepUpdate("Fetching PR #$prNumber from $repo...")
        val pr = client.getPR(repo, prNumber)
        val files = client.getPRFiles(repo, prNumber)

        val diffInput = buildDiffInput(pr, files)

        val profile = ProfileLoader.load("github-pr-review")
            ?: error("Profile 'github-pr-review' not found. Make sure profiles/github-pr-review.json exists.")

        val result = engine.execute(
            config  = profile,
            input   = diffInput,
            context = emptyMap(),
            onStepStart = { index, total, step ->
                onStepUpdate("[Step $index/$total] ${step.name}...")
            },
            onStepComplete = { index, total, stepResult ->
                onStepUpdate(" done (${stepResult.provider} / ${stepResult.model})")
            }
        )

        val (reviewBody, inlineComments) = parseInlineComments(result.finalOutput)

        onStepUpdate("Posting review to GitHub...")
        return try {
            client.postReview(
                repo       = repo,
                prNumber   = prNumber,
                commitSha  = pr.head.sha,
                body       = reviewBody,
                event      = "COMMENT",
                comments   = inlineComments
            )
            "Review posted to ${pr.htmlUrl}"
        } catch (e: Exception) {
            onStepUpdate("Review API failed (${e.message}), falling back to issue comment...")
            client.postIssueComment(repo, prNumber, reviewBody)
            "Comment posted to ${pr.htmlUrl} (review API fallback)"
        }
    }

    // -------------------------------------------------------------------------
    // PR Migration
    // -------------------------------------------------------------------------

    /**
     * Fetches PR files, runs the `code-migration` orchestration profile on the
     * file contents, and posts the migration output as an issue comment.
     *
     * @param repo          "owner/repo" format
     * @param prNumber      Pull request number
     * @param fromLang      Source language (e.g. "Java")
     * @param toLang        Target language (e.g. "Kotlin")
     * @param onStepUpdate  Called before each orchestration step with a status message
     */
    fun migratePR(
        repo: String,
        prNumber: Int,
        fromLang: String,
        toLang: String,
        onStepUpdate: (String) -> Unit = {}
    ): String {
        onStepUpdate("Fetching PR #$prNumber from $repo...")
        val pr = client.getPR(repo, prNumber)
        val files = client.getPRFiles(repo, prNumber)

        onStepUpdate("Fetching file contents (${files.size} file(s))...")
        val fileContents = buildFileContents(repo, files, pr.head.ref)

        val profile = ProfileLoader.load("code-migration")
            ?: error("Profile 'code-migration' not found. Make sure profiles/code-migration.json exists.")

        val result = engine.execute(
            config  = profile,
            input   = fileContents,
            context = mapOf("from" to fromLang, "to" to toLang),
            onStepStart = { index, total, step ->
                onStepUpdate("[Step $index/$total] ${step.name}...")
            },
            onStepComplete = { index, total, stepResult ->
                onStepUpdate(" done (${stepResult.provider} / ${stepResult.model})")
            }
        )

        onStepUpdate("Posting migration result to GitHub...")
        client.postIssueComment(repo, prNumber, result.finalOutput)
        return "Migration comment posted to ${pr.htmlUrl}"
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private fun buildDiffInput(pr: PullRequest, files: List<PullRequestFile>): String {
        val sb = StringBuilder()
        sb.appendLine("# PR #${pr.number}: ${pr.title}")
        if (!pr.body.isNullOrBlank()) {
            sb.appendLine()
            sb.appendLine("## Description")
            sb.appendLine(pr.body)
        }
        sb.appendLine()
        sb.appendLine("## Changed Files (${files.size})")
        for (file in files) {
            sb.appendLine()
            sb.appendLine("### ${file.filename} [${file.status}] (+${file.additions}/-${file.deletions})")
            if (file.patch != null) {
                sb.appendLine("```diff")
                sb.appendLine(file.patch)
                sb.appendLine("```")
            } else {
                sb.appendLine("*(binary or no patch available)*")
            }
        }
        return sb.toString()
    }

    private fun buildFileContents(repo: String, files: List<PullRequestFile>, ref: String): String {
        val sb = StringBuilder()
        val filesToFetch = if (files.size <= 3) files else files.take(3)
        for (file in filesToFetch) {
            sb.appendLine("### File: ${file.filename}")
            sb.appendLine()
            try {
                val content = client.getFileContent(repo, file.filename, ref)
                sb.appendLine("```")
                sb.appendLine(content)
                sb.appendLine("```")
            } catch (e: Exception) {
                // Fall back to patch if full content fetch fails
                if (file.patch != null) {
                    sb.appendLine("*(full content unavailable, showing diff)*")
                    sb.appendLine("```diff")
                    sb.appendLine(file.patch)
                    sb.appendLine("```")
                } else {
                    sb.appendLine("*(content unavailable)*")
                }
            }
            sb.appendLine()
        }
        if (files.size > 3) {
            sb.appendLine("*(${files.size - 3} additional file(s) omitted for brevity)*")
        }
        return sb.toString()
    }

    /**
     * Extracts the `<INLINE_COMMENTS>[...]</INLINE_COMMENTS>` block from [text].
     * Returns a pair of (bodyWithoutBlock, parsedComments).
     * If the block is missing or malformed, returns (originalText, emptyList).
     */
    fun parseInlineComments(text: String): Pair<String, List<InlineComment>> {
        val regex = Regex("""<INLINE_COMMENTS>(\[.*?])</INLINE_COMMENTS>""", RegexOption.DOT_MATCHES_ALL)
        val match = regex.find(text) ?: return Pair(text, emptyList())

        val body = text.removeRange(match.range).trim()
        val jsonArray = match.groupValues[1]

        return try {
            val gson = com.google.gson.GsonBuilder()
                .setFieldNamingPolicy(com.google.gson.FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
                .create()
            val type = object : com.google.gson.reflect.TypeToken<List<InlineComment>>() {}.type
            val comments: List<InlineComment> = gson.fromJson(jsonArray, type) ?: emptyList()
            Pair(body, comments)
        } catch (e: Exception) {
            // JSON was malformed — return body without the block but no comments
            Pair(body, emptyList())
        }
    }
}
