package cli

import ai.ChatMessage
import ai.ProviderRegistry
import ai.orchestration.OrchestrationEngine
import ai.orchestration.ProfileLoader
import ai.router.ModelRouter
import github.GitHubReviewOrchestrator

class SingleShotCLI {
    fun run(args: Array<String>) {
        printProviderStatus()
        val parsed = parseArgs(args)

        // ── GitHub PR review mode ─────────────────────────────────────────────
        val githubReviewRepo = parsed["github-review"]
        if (githubReviewRepo != null) {
            val prNumber = parsed["pr"]?.toIntOrNull()
                ?: error("--github-review requires --pr <number>")
            val orchestrator = GitHubReviewOrchestrator()
            val outcome = orchestrator.reviewPR(
                repo        = githubReviewRepo,
                prNumber    = prNumber,
                onStepUpdate = { msg -> System.err.print(msg) }
            )
            System.err.println()
            println(outcome)
            return
        }

        // ── GitHub PR migration mode ───────────────────────────────────────────
        val githubMigrateRepo = parsed["github-migrate"]
        if (githubMigrateRepo != null) {
            val prNumber = parsed["pr"]?.toIntOrNull()
                ?: error("--github-migrate requires --pr <number>")
            val fromLang = parsed["from"] ?: error("--github-migrate requires --from <language>")
            val toLang   = parsed["to"]   ?: error("--github-migrate requires --to <language>")
            val orchestrator = GitHubReviewOrchestrator()
            val outcome = orchestrator.migratePR(
                repo        = githubMigrateRepo,
                prNumber    = prNumber,
                fromLang    = fromLang,
                toLang      = toLang,
                onStepUpdate = { msg -> System.err.print(msg) }
            )
            System.err.println()
            println(outcome)
            return
        }

        val prompt = parsed["prompt"]
            ?: error("Missing prompt. Provide your prompt as the last argument.")
        val messages = listOf(ChatMessage(role = "user", content = prompt))

        // ── Profile mode ──────────────────────────────────────────────────────
        val profileName = parsed["profile"]
        if (profileName != null) {
            val config = ProfileLoader.load(profileName)
                ?: error("Profile '$profileName' not found. Add a JSON file to the profiles/ directory.")

            // Build context from CLI flags and any matching parsed keys
            val context = mutableMapOf<String, String>()
            parsed["from"]?.let { context["from"] = it }
            parsed["to"]?.let   { context["to"]   = it }

            // Validate that all required context prompts were supplied
            val missing = config.contextPrompts
                ?.filter { cp -> cp.key !in context }
                ?.map    { "--${it.key}" }
                ?: emptyList()
            if (missing.isNotEmpty()) {
                error("Profile '${config.name}' requires: ${missing.joinToString()}")
            }

            val engine = OrchestrationEngine()
            val result = engine.execute(
                config         = config,
                input          = prompt,
                context        = context,
                onStepStart    = { index, total, step ->
                    System.err.print("[Step $index/$total] ${step.name}...")
                    System.err.flush()
                },
                onStepComplete = { _, _, stepResult ->
                    System.err.println(" done (${stepResult.provider} / ${stepResult.model})")
                }
            )

            println(result.finalOutput)
            return
        }

        // ── Auto-routing mode ─────────────────────────────────────────────────
        if (parsed.containsKey("auto")) {
            val router = ModelRouter()
            val (response, decision) = router.routeAndChat(messages)
            System.err.println("[Auto-routed: ${decision.taskType.displayName} → ${decision.provider.name} / ${decision.model}]")
            println(response.content)
            return
        }

        // ── Explicit provider / model mode ────────────────────────────────────
        val providerName = parsed["provider"]
            ?: error("Missing --provider flag. Usage: --provider <name> --model <model> <prompt>  (or --auto / --profile <name>)")
        val model = parsed["model"]
            ?: error("Missing --model flag. Usage: --provider <name> --model <model> <prompt>  (or --auto / --profile <name>)")

        val provider = ProviderRegistry.getByName(providerName)
            ?: error("Unknown provider: '$providerName'. Available: ${ProviderRegistry.availableProviders().joinToString { it.name }}")

        if (!provider.isAvailable()) {
            error("Provider '$providerName' is not available. Check your API key or connection.")
        }

        val response = provider.chat(messages, model)
        println(response.content)
    }

    private fun printProviderStatus() {
        System.err.println("Model sources:")
        ProviderRegistry.allProviders().forEach { p ->
            if (p.isAvailable()) {
                val models = p.availableModels.joinToString(", ")
                System.err.println("  [OK] %-12s %s".format(p.name, models))
            } else {
                System.err.println("  [--] %-12s not available".format(p.name))
            }
        }
        System.err.println()
    }

    private fun parseArgs(args: Array<String>): Map<String, String> {
        val result = mutableMapOf<String, String>()
        var i = 0
        while (i < args.size) {
            when (args[i]) {
                "--auto"           -> { result["auto"]           = "true";                      i++ }
                "--profile"        -> { result["profile"]        = args.getOrNull(++i) ?: "";   i++ }
                "--from"           -> { result["from"]           = args.getOrNull(++i) ?: "";   i++ }
                "--to"             -> { result["to"]             = args.getOrNull(++i) ?: "";   i++ }
                "--provider"       -> { result["provider"]       = args.getOrNull(++i) ?: "";   i++ }
                "--model"          -> { result["model"]          = args.getOrNull(++i) ?: "";   i++ }
                "--github-review"  -> { result["github-review"]  = args.getOrNull(++i) ?: "";   i++ }
                "--github-migrate" -> { result["github-migrate"] = args.getOrNull(++i) ?: "";   i++ }
                "--pr"             -> { result["pr"]             = args.getOrNull(++i) ?: "";   i++ }
                else -> {
                    if (!args[i].startsWith("--")) {
                        val prompt = args.drop(i).joinToString(" ")
                            .removeSurrounding("'").removeSurrounding("\"")
                        result["prompt"] = prompt
                        break
                    }
                    i++
                }
            }
        }
        return result
    }
}
