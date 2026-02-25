package cli

import ai.AIProvider
import ai.ChatMessage
import ai.ProviderRegistry
import ai.orchestration.OrchestrationEngine
import ai.orchestration.ProfileLoader
import ai.providers.OllamaModelCatalog
import ai.providers.OllamaProvider
import ai.router.ModelRouter

class InteractiveCLI {
    private val router  = ModelRouter()
    private val engine  = OrchestrationEngine()
    private val history = mutableListOf<ChatMessage>()

    fun run() {
        println("=== AgenticAI Sandbox ===")
        printProviderStatus()

        val providers = ProviderRegistry.availableProviders()
        if (providers.isEmpty()) {
            println("No providers available. Set API keys in your environment or .env file (see .env.example).")
            return
        }

        var autoMode = false
        var provider = selectProvider(providers) ?: return
        var model    = selectModel(provider)     ?: return

        println("\nStarting conversation with ${provider.name} / $model")
        println("Commands: /switch, /auto, /profile [name], /clear, /exit")
        println()

        while (true) {
            print("You: ")
            val input = readLine()?.trim() ?: break

            // Prefix command: /profile [name]
            if (input.lowercase().startsWith("/profile")) {
                handleProfile(input.drop("/profile".length).trim())
                continue
            }

            when (input.lowercase()) {
                "/exit" -> {
                    println("Goodbye!")
                    break
                }
                "" -> continue
                "/clear" -> {
                    history.clear()
                    println("[Conversation history cleared]")
                    continue
                }
                "/auto" -> {
                    autoMode = !autoMode
                    if (autoMode) {
                        println("[Auto-routing enabled — provider and model chosen automatically per message]")
                    } else {
                        println("[Auto-routing disabled — using ${provider.name} / $model]")
                    }
                    continue
                }
                "/switch" -> {
                    val newProviders = ProviderRegistry.availableProviders()
                    val newProvider  = selectProvider(newProviders) ?: continue
                    val newModel     = selectModel(newProvider)     ?: continue
                    provider  = newProvider
                    model     = newModel
                    autoMode  = false
                    history.clear()
                    println("[Switched to ${provider.name} / $model — auto-routing off, history cleared]")
                    continue
                }
                else -> {
                    history.add(ChatMessage(role = "user", content = input))
                    try {
                        if (autoMode) {
                            val (response, decision) = router.routeAndChat(history)
                            history.add(ChatMessage(role = "assistant", content = response.content))
                            println("\n[Auto-routed: ${decision.taskType.displayName} → ${decision.provider.name} / ${decision.model}]")
                            println("${decision.provider.name} (${decision.model}): ${response.content}\n")
                        } else {
                            val response = provider.chat(history, model)
                            history.add(ChatMessage(role = "assistant", content = response.content))
                            println("\n${provider.name} ($model): ${response.content}\n")
                        }
                    } catch (e: Exception) {
                        println("[Error: ${e.message}]")
                        history.removeLastOrNull()
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Profile orchestration
    // -------------------------------------------------------------------------

    private fun handleProfile(arg: String) {
        if (arg.isBlank()) {
            val profiles = ProfileLoader.listAvailable()
            if (profiles.isEmpty()) {
                println("[No profiles found. Add JSON files to the profiles/ directory.]")
            } else {
                println("\nAvailable profiles:")
                profiles.forEach { (name, desc) -> println("  %-22s %s".format(name, desc)) }
                println("\nUsage: /profile <name>")
            }
            return
        }

        val config = ProfileLoader.load(arg)
        if (config == null) {
            println("[Profile '$arg' not found. Use /profile to list available profiles.]")
            return
        }

        println("\n[Profile: ${config.name}]")
        println(config.description)
        println()

        // Collect context values required by the profile (e.g., "from", "to")
        val context = mutableMapOf<String, String>()
        for (cp in config.contextPrompts ?: emptyList()) {
            print("${cp.prompt} ")
            val value = readLine()?.trim()
            if (value.isNullOrBlank()) {
                println("[Cancelled]")
                return
            }
            context[cp.key] = value
        }

        // Collect multi-line main input
        val input = readMultilineInput() ?: return
        if (input.isBlank()) {
            println("[No input provided — cancelled.]")
            return
        }

        println()
        try {
            val result = engine.execute(
                config          = config,
                input           = input,
                context         = context,
                onStepStart     = { index, total, step ->
                    print("  [Step $index/$total] ${step.name}...")
                    System.out.flush()
                },
                onStepComplete  = { _, _, stepResult ->
                    println(" done  (${stepResult.provider} / ${stepResult.model})")
                }
            )

            println("\n=== ${config.name} — Result ===\n")
            println(result.finalOutput)
            println()

            // Append the profile run to conversation history so follow-up questions work
            history.add(ChatMessage("user",      "[Profile: ${config.name}]\n$input"))
            history.add(ChatMessage("assistant", result.finalOutput))
        } catch (e: Exception) {
            println()
            println("[Profile error: ${e.message}]")
        }
    }

    /** Reads lines until the user types END (case-insensitive) on its own line. */
    private fun readMultilineInput(): String? {
        println("Enter input (paste code or text, then type END on a new line to finish):")
        val lines = mutableListOf<String>()
        while (true) {
            val line = readLine() ?: return null
            if (line.trim().uppercase() == "END") break
            lines.add(line)
        }
        return lines.joinToString("\n")
    }

    // -------------------------------------------------------------------------
    // Provider / model selection
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Provider status
    // -------------------------------------------------------------------------

    private fun printProviderStatus() {
        println("\nModel sources:")
        ProviderRegistry.allProviders().forEach { p ->
            if (p.isAvailable()) {
                val models = p.availableModels.joinToString(", ")
                println("  [OK] %-12s %s".format(p.name, models))
            } else {
                println("  [--] %-12s not available".format(p.name))
            }
        }
        println()
    }

    private fun selectProvider(providers: List<AIProvider>): AIProvider? {
        println("\nAvailable providers:")
        providers.forEachIndexed { i, p -> println("  ${i + 1}. ${p.name}") }
        print("Select provider (number): ")
        val input = readLine()?.trim() ?: return null
        val index = input.toIntOrNull()?.minus(1) ?: return null
        return providers.getOrNull(index).also {
            if (it == null) println("Invalid selection.")
        }
    }

    private fun selectModel(provider: AIProvider): String? {
        if (provider is OllamaProvider) return selectOllamaModel(provider)

        val models = provider.availableModels
        if (models.isEmpty()) {
            println("No models available for ${provider.name}.")
            return null
        }
        println("\nAvailable models for ${provider.name}:")
        models.forEachIndexed { i, m -> println("  ${i + 1}. $m") }
        print("Select model (number): ")
        val input = readLine()?.trim() ?: return null
        val index = input.toIntOrNull()?.minus(1) ?: return null
        return models.getOrNull(index).also {
            if (it == null) println("Invalid selection.")
        }
    }

    /**
     * Ollama-specific model selector. Shows installed models first, then
     * recommended catalog models not yet installed, with VRAM and task hints.
     * Pre-selects the first installed model (or first catalog default if none installed).
     */
    private fun selectOllamaModel(provider: OllamaProvider): String? {
        val installed = provider.availableModels

        // Catalog split: defaults not yet installed, then remaining catalog
        val catalogNotInstalled = OllamaModelCatalog.models.filter { it.id !in installed }
        val defaultsNotInstalled = catalogNotInstalled.filter { it.isDefault }
        val otherNotInstalled    = catalogNotInstalled.filter { !it.isDefault }

        // Build flat entry list for numbered selection
        data class Entry(val id: String, val isInstalled: Boolean)
        val entries = mutableListOf<Entry>()

        println("\nModels for ollama:")

        if (installed.isNotEmpty()) {
            println("  Installed:")
            installed.forEach { id ->
                val num = entries.size + 1
                println("    %2d. %s".format(num, id))
                entries.add(Entry(id, true))
            }
        }

        if (defaultsNotInstalled.isNotEmpty()) {
            println("  Recommended (not installed — run: ollama pull <model>):")
            defaultsNotInstalled.forEach { info ->
                val num = entries.size + 1
                val tasks = info.taskTypes.joinToString(", ") { it.replaceFirstChar(Char::uppercaseChar) }
                val sizeLabel = when {
                    info.minVramGb <= 3  -> "~${info.minVramGb}GB VRAM / CPU-friendly"
                    info.minVramGb <= 6  -> "~${info.minVramGb}GB VRAM"
                    else                 -> "~${info.minVramGb}GB VRAM"
                }
                println("    %2d. ★ %-28s %.0fB · %s · %s".format(
                    num, info.id, info.paramsBillions, sizeLabel, tasks))
                println("        ${info.description}")
                entries.add(Entry(info.id, false))
            }
        }

        if (otherNotInstalled.isNotEmpty()) {
            println("  More available (not installed):")
            otherNotInstalled.forEach { info ->
                val num = entries.size + 1
                val tasks = info.taskTypes.joinToString(", ") { it.replaceFirstChar(Char::uppercaseChar) }
                println("    %2d.   %-28s %.0fB · ~%dGB VRAM · %s".format(
                    num, info.id, info.paramsBillions, info.minVramGb, tasks))
                entries.add(Entry(info.id, false))
            }
        }

        if (entries.isEmpty()) {
            println("  No models available. Start Ollama and pull a model:")
            OllamaModelCatalog.defaults.forEach { println("    ollama pull ${it.id}") }
            return null
        }

        val defaultEntry = entries.firstOrNull { it.isInstalled } ?: entries.first()
        val defaultNum   = entries.indexOf(defaultEntry) + 1

        print("\nSelect model (number) [default: $defaultNum]: ")
        val input = readLine()?.trim() ?: return null

        val entry = if (input.isEmpty()) {
            defaultEntry
        } else {
            val idx = input.toIntOrNull()?.minus(1) ?: run {
                println("Invalid selection.")
                return null
            }
            entries.getOrNull(idx) ?: run {
                println("Invalid selection.")
                return null
            }
        }

        if (!entry.isInstalled) {
            println("  Note: '${entry.id}' is not installed. Pull it first with: ollama pull ${entry.id}")
        }
        return entry.id
    }
}
