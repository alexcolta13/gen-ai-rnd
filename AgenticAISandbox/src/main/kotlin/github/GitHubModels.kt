package github

data class BranchRef(
    val ref: String,
    val sha: String
)

data class PullRequest(
    val number: Int,
    val title: String,
    val body: String?,
    val htmlUrl: String,
    val head: BranchRef,
    val base: BranchRef
)

data class PullRequestFile(
    val filename: String,
    val status: String,
    val additions: Int,
    val deletions: Int,
    val patch: String?
)

data class InlineComment(
    val path: String,
    val line: Int,
    val body: String,
    val side: String = "RIGHT"
)
