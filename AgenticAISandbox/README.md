# AgenticAI Sandbox

A unified CLI for interacting with multiple AI providers — **OpenAI**, **Anthropic**, and **Ollama** — from a single interface. Supports interactive REPL conversations, single-shot scripting, keyword-based automatic routing, and multi-step orchestration profiles.

Built with **Kotlin 2.0** and **Maven**.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Providers](#providers)
- [Usage](#usage)
  - [Interactive Mode](#interactive-mode)
  - [Single-Shot Mode](#single-shot-mode)
  - [Auto-Routing](#auto-routing)
  - [Orchestration Profiles](#orchestration-profiles)
- [GitHub Integration](#github-integration)
- [MCP Server](#mcp-server)
- [Profile JSON Reference](#profile-json-reference)
- [Built-in Profiles](#built-in-profiles)
- [Creating Custom Profiles](#creating-custom-profiles)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Features

- **Unified provider interface** — OpenAI, Anthropic, and Ollama behind a single `AIProvider` abstraction
- **Interactive REPL** — persistent conversation history with in-session `/switch`, `/auto`, `/profile`, `/clear`, `/exit` commands
- **Single-shot CLI** — pipe-friendly one-liner mode; all status output goes to stderr, result to stdout
- **Auto-routing** — keyword classification engine automatically picks the best provider and model per task type with a fallback chain
- **Multi-step orchestration profiles** — JSON-defined pipelines where each step's output feeds the next via `{{variable}}` templates
- **Per-profile model preferences** — ordered preference lists at the profile level and per-step level, with automatic fallback through unavailable providers
- **Dynamic Ollama model discovery** — installed models fetched live from the Ollama API at startup
- **Startup provider report** — shows which providers are configured and available before any interaction

---

## Architecture

```
Main.kt                     Entry point → InteractiveCLI or SingleShotCLI
McpMain.kt                  MCP server entry point → GitHubMcpServer
├── InteractiveCLI          REPL loop, conversation history, command dispatch
└── SingleShotCLI           Argument parser, one-shot + GitHub flags

ai/
├── AIProvider              Interface: name, availableModels, chat(), isAvailable()
├── ProviderRegistry        Holds all providers; gates on isAvailable()
├── providers/
│   ├── OpenAIProvider      POST /v1/chat/completions
│   ├── AnthropicProvider   POST /v1/messages  (separates system turns)
│   └── OllamaProvider      GET /api/tags (model list) · POST /api/chat
├── router/
│   ├── TaskType            CODING · ANALYSIS · CREATIVE · SIMPLE · GENERAL
│   ├── RouteTarget         (providerName, model) pair
│   └── ModelRouter         Classify → priority fallback chain → chat
└── orchestration/
    ├── ProfileConfig       Data classes: ProfileConfig, StepConfig,
    │                       ModelPreferences, ModelPreference, ContextPrompt
    ├── OrchestrationResult StepResult, OrchestrationResult
    ├── ProfileLoader       Loads/lists profiles/*.json via Gson
    └── OrchestrationEngine Renders {{templates}}, resolves models, runs steps

github/
├── GitHubModels            PullRequest, BranchRef, PullRequestFile, InlineComment
├── GitHubClient            OkHttp REST client for GitHub API v3
└── GitHubReviewOrchestrator Fetch PR → run profile → post review/comment

mcp/
├── McpServer               Abstract JSON-RPC 2.0 stdio server (protocol 2024-11-05)
└── GitHubMcpServer         4 tools: get_pr, post_review, post_comment, get_file

config/
└── Config                  Env var → .env fallback (dotenv-kotlin)

profiles/
├── code-review.json        4-step local code review pipeline
├── code-migration.json     4-step code migration pipeline
└── github-pr-review.json   3-step GitHub PR review pipeline
```

### Model Resolution Order

Every step in a profile resolves its provider and model through three levels:

```
1. stepOverrides[stepId]   ordered list — first available provider+model wins
2. default                 ordered list — first available provider+model wins
3. ModelRouter             keyword classification + routing table fallback chain
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| JDK | 8 or later |
| Maven | 3.6 or later |
| Ollama *(optional)* | Any — must be running locally |

At least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` must be set, or Ollama must be reachable, to use any feature.

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd AgenticAISandbox

# 2. Configure credentials
cp .env.example .env
# Edit .env — fill in API keys for OpenAI, Anthropic, and/or GITHUB_TOKEN

# 3. Build
mvn compile

# 4. Run (interactive REPL)
mvn exec:java

# 5. (Optional) Start the GitHub MCP server
mvn exec:java@mcp
```

---

## Configuration

Copy `.env.example` to `.env` and fill in values. Environment variables take precedence over `.env`.

```dotenv
# OpenAI — https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# Anthropic — https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=sk-ant-...

# Ollama (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434
```

| Variable | Required | Default |
|---|---|---|
| `OPENAI_API_KEY` | If using OpenAI | — |
| `ANTHROPIC_API_KEY` | If using Anthropic | — |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` |
| `GITHUB_TOKEN` | For GitHub integration | — |

`.env` is git-ignored. Never commit real credentials.

---

## Providers

### OpenAI

| Model | Notes |
|---|---|
| `gpt-4o` | Default for general, creative, and coding tasks |
| `gpt-4o-mini` | Default for simple/quick tasks |
| `gpt-3.5-turbo` | Lightweight fallback |

**Available when:** `OPENAI_API_KEY` is set.

---

### Anthropic

| Model | Notes |
|---|---|
| `claude-opus-4-6` | Highest capability; default for analysis tasks |
| `claude-sonnet-4-6` | Balanced; default for coding and general tasks |
| `claude-haiku-3-5` | Fast and lightweight |

**Available when:** `ANTHROPIC_API_KEY` is set.

---

### Ollama

Installed models are fetched dynamically from `GET /api/tags` at startup. The sandbox ships a **built-in catalog** of 14 curated free models tested on consumer CUDA GPUs and Apple Silicon (Metal). The catalog is shown in the model selector — installed models appear first, recommended models (★) and the rest of the catalog appear below with VRAM requirements and task hints.

**Available when:** Ollama is running and reachable at `OLLAMA_BASE_URL`.

#### Recommended models (★ defaults)

| Model | Params | Min VRAM | Best for | Pull command |
|---|---|---|---|---|
| `llama3.2:3b` | 3B | 2 GB | General, Simple | `ollama pull llama3.2:3b` |
| `llama3.1:8b` | 8B | 5 GB | General, Coding, Analysis | `ollama pull llama3.1:8b` |
| `qwen2.5-coder:7b` | 7B | 5 GB | Coding | `ollama pull qwen2.5-coder:7b` |
| `phi4:14b` | 14B | 9 GB | Coding, Analysis, Reasoning | `ollama pull phi4:14b` |

#### Full model catalog

| Model | Params | Min VRAM | Best for |
|---|---|---|---|
| `llama3.2:3b` ★ | 3B | 2 GB | General, Simple |
| `phi4-mini:3.8b` | 3.8B | 3 GB | Reasoning, Simple |
| `qwen2.5:3b` | 3B | 2 GB | General (multilingual) |
| `llama3.1:8b` ★ | 8B | 5 GB | General, Coding, Analysis |
| `qwen2.5-coder:7b` ★ | 7B | 5 GB | Coding |
| `mistral:7b` | 7B | 5 GB | General, Creative |
| `deepseek-r1:7b` | 7B | 5 GB | Analysis, Reasoning |
| `gemma2:9b` | 9B | 6 GB | General, Analysis |
| `codellama:7b` | 7B | 5 GB | Coding |
| `phi4:14b` ★ | 14B | 9 GB | Coding, Analysis, Reasoning |
| `qwen2.5-coder:14b` | 14B | 9 GB | Coding |
| `deepseek-r1:14b` | 14B | 9 GB | Analysis, Reasoning |
| `mistral-small:22b` | 22B | 14 GB | General, Analysis, Creative |
| `qwen2.5:32b` | 32B | 20 GB | General, Analysis |

★ = included in the recommended defaults shown in the model selector.

`minVramGb` is the practical minimum for Q4-quantised GPU inference. All models run on CPU-only machines — just slower.

---

## Usage

### Interactive Mode

```bash
mvn exec:java
```

On startup, a provider status table is printed:

```
=== AgenticAI Sandbox ===

Model sources:
  [OK] openai       gpt-4o, gpt-4o-mini, gpt-3.5-turbo
  [OK] anthropic    claude-opus-4-6, claude-sonnet-4-6, claude-haiku-3-5
  [--] ollama       not available
```

Select a provider and model, then start chatting. Conversation history is maintained for the session.

**REPL commands:**

| Command | Description |
|---|---|
| `/switch` | Switch to a different provider and model (clears history) |
| `/auto` | Toggle auto-routing — provider and model chosen automatically per message |
| `/profile [name]` | Run an orchestration profile; omit name to list available profiles |
| `/clear` | Clear conversation history |
| `/exit` | Quit |

---

### Single-Shot Mode

Pass arguments directly for non-interactive use. All status output goes to **stderr**; only the AI response is written to **stdout**, making it safe to pipe.

```bash
# Explicit provider and model
mvn exec:java -Dexec.args="--provider anthropic --model claude-sonnet-4-6 Explain tail recursion"

# Auto-routing
mvn exec:java -Dexec.args="--auto Write a Python function to parse ISO 8601 dates"

# Profile with context variables
mvn exec:java -Dexec.args="--profile code-migration --from Java/Spring --to Kotlin/Ktor < MyService.java"

# Pipe the result to a file
mvn exec:java -Dexec.args="--auto Summarize REST API design principles" > summary.txt
```

**Flags:**

| Flag | Description |
|---|---|
| `--provider <name>` | Provider name: `openai`, `anthropic`, `ollama` |
| `--model <model>` | Exact model name (required with `--provider`) |
| `--auto` | Auto-route based on prompt content |
| `--profile <name>` | Run an orchestration profile by filename (without `.json`) |
| `--from <value>` | Context variable for profiles that require a source language/framework |
| `--to <value>` | Context variable for profiles that require a target language/framework |

---

### Auto-Routing

When `--auto` is used (single-shot) or `/auto` is toggled on (REPL), the `ModelRouter` classifies the prompt using keyword heuristics and dispatches to the best available provider, falling back through the chain if a provider is unavailable or the API call fails.

**Routing table** (first available provider+model wins; each row lists the full fallback chain):

| Task Type | Trigger | Cloud chain | Ollama fallback chain |
|---|---|---|---|
| `CODING` | Code/syntax keywords (≥2 matches) | anthropic/claude-sonnet-4-6 → openai/gpt-4o | qwen2.5-coder:7b → phi4:14b → qwen2.5-coder:14b → codellama:7b → deepseek-r1:7b → llama3.1:8b → any |
| `ANALYSIS` | Analyze/compare/explain or long prompts (>80 words) | anthropic/claude-opus-4-6 → openai/gpt-4o → anthropic/claude-sonnet-4-6 | deepseek-r1:7b → phi4:14b → deepseek-r1:14b → gemma2:9b → llama3.1:8b → any |
| `CREATIVE` | Story/poem/creative writing keywords | openai/gpt-4o → anthropic/claude-sonnet-4-6 | llama3.1:8b → mistral:7b → mistral-small:22b → any |
| `SIMPLE` | Short prompts (≤15 words) or factual starters | openai/gpt-4o-mini → anthropic/claude-haiku-3-5 | llama3.2:3b → phi4-mini:3.8b → qwen2.5:3b → llama3.1:8b → any |
| `GENERAL` | Everything else | anthropic/claude-sonnet-4-6 → openai/gpt-4o | llama3.1:8b → phi4:14b → llama3.2:3b → mistral:7b → any |

Ollama entries only match if the model is installed (`ollama pull <model>` first). The bare `any` fallback matches whatever is installed.

The auto-routing decision is shown inline:

```
[Auto-routed: Coding / Technical → anthropic / claude-sonnet-4-6]
```

---

### Orchestration Profiles

Profiles are multi-step pipelines defined in JSON files under `profiles/`. Each step sends a prompt to an AI model, and its output is injected as a `{{variable}}` into subsequent steps.

**Run in interactive mode:**
```
/profile code-review
```

**Run in single-shot mode:**
```bash
mvn exec:java -Dexec.args="--profile code-review" < src/MyClass.kt
```

**Step progress is printed as each step completes:**
```
  [Step 1/4] Bug & Logic Analysis... done  (anthropic / claude-opus-4-6)
  [Step 2/4] Security Audit... done  (anthropic / claude-opus-4-6)
  [Step 3/4] Style & Best Practices... done  (anthropic / claude-opus-4-6)
  [Step 4/4] Review Summary Report... done  (anthropic / claude-opus-4-6)
```

---

## GitHub Integration

AgenticAI Sandbox can fetch pull requests directly from GitHub, run AI analysis via the orchestration engine, and post results back as PR reviews or comments.

### Setup

Add your GitHub personal access token (needs `repo` and `pull_requests` scopes) to `.env`:

```dotenv
GITHUB_TOKEN=ghp_...
```

### PR Review

Fetches the PR diff, runs the `github-pr-review` 3-step profile, and posts a formal review with inline comments:

```bash
mvn exec:java -Dexec.args="--github-review owner/repo --pr 123"
```

Steps: **PR Diff Analysis** → **Security Scan** → **Review Synthesis** (produces review body + inline comments)

Status output goes to **stderr**; the review URL goes to **stdout**.

### PR Migration

Fetches file contents from a PR, runs the `code-migration` profile, and posts the migrated code as a comment:

```bash
mvn exec:java -Dexec.args="--github-migrate owner/repo --pr 123 --from Java --to Kotlin"
```

### New Flags

| Flag | Description |
|---|---|
| `--github-review <owner/repo>` | Run AI review on a GitHub PR and post results |
| `--github-migrate <owner/repo>` | Run code migration on a GitHub PR and post result comment |
| `--pr <number>` | Pull request number (required with `--github-review` and `--github-migrate`) |
| `--from <language>` | Source language for migration |
| `--to <language>` | Target language for migration |

---

## MCP Server

The project ships a standalone **Model Context Protocol (MCP)** server (`McpMain.kt`) that exposes GitHub operations as tools for any MCP-compatible client (Claude Desktop, Cursor, etc.).

The server uses **JSON-RPC 2.0 over stdio** and implements protocol version `2024-11-05`.

### Start the server

```bash
mvn exec:java@mcp
```

### Available tools

| Tool | Arguments | Description |
|---|---|---|
| `github_get_pr` | `repo`, `pr` | Fetch PR metadata, description, and file diffs |
| `github_post_review` | `repo`, `pr`, `body`, `event`, `comments?` | Post a formal PR review with optional inline comments |
| `github_post_comment` | `repo`, `pr`, `body` | Post a plain issue comment on a PR |
| `github_get_file` | `repo`, `path`, `ref` | Retrieve decoded file content at a specific ref |

### Claude Desktop configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "github": {
      "command": "mvn",
      "args": ["exec:java@mcp", "-f", "/path/to/AgenticAISandbox/pom.xml"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    }
  }
}
```

### Testing manually

```bash
# In one terminal, start the server
mvn exec:java@mcp

# In another, send a JSON-RPC request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | mvn exec:java@mcp
```

---

## Profile JSON Reference

```jsonc
{
  "name": "profile-name",
  "description": "Human-readable description shown in /profile list",

  // Variables the user is prompted for before the profile runs.
  // Each collected value is injected as {{key}} in all step templates.
  "contextPrompts": [
    { "key": "from", "prompt": "Source language/framework:" },
    { "key": "to",   "prompt": "Target language/framework:" }
  ],

  "steps": [
    {
      "id": "step-id",           // Unique identifier; used in stepOverrides and as default outputKey
      "name": "Display Name",    // Shown in progress output
      "systemPrompt": "...",     // System role instructions (supports {{variable}} templates)
      "userPromptTemplate": "...",// User turn prompt (supports {{variable}} templates)
      "outputKey": "step-id"     // Optional; defaults to id. Use as {{step_<outputKey>}} in later steps
    }
  ],

  // Model preferences for this profile.
  // Omit entirely to fall through to ModelRouter for every step.
  "modelPreferences": {

    // Ordered list — tried in sequence, first available provider+model wins.
    // Applies to all steps that have no entry in stepOverrides.
    "default": [
      { "provider": "anthropic", "model": "claude-opus-4-6" },
      { "provider": "openai",    "model": "gpt-4o" }
    ],

    // Per-step overrides. Each value is also an ordered preference list.
    // An entry here takes priority over "default" for that step only.
    "stepOverrides": {
      "step-id": [
        { "provider": "anthropic", "model": "claude-sonnet-4-6" },
        { "provider": "openai",    "model": "gpt-4o-mini" }
      ]
    }
  }
}
```

### Built-in Template Variables

| Variable | Available in | Value |
|---|---|---|
| `{{input}}` | All steps | The user's original input text |
| `{{step_<outputKey>}}` | Steps after the one that produced it | Output of the named step |
| `{{from}}`, `{{to}}` | All steps (if defined in contextPrompts) | User-supplied context values |

### Model Preference Resolution

For each step, the engine walks through three levels and uses the first that resolves to an available provider and registered model:

```
stepOverrides[stepId][0], [1], ... → default[0], [1], ... → ModelRouter
```

If a provider is unavailable or a model name is not in its model list, that entry is silently skipped and the next is tried.

---

## Built-in Profiles

### `code-review`

Four-step pipeline for thorough code review. Accepts any code snippet as input.

| Step | Role | Task |
|---|---|---|
| `bug-analysis` | Senior engineer | Bugs, logic errors, null risks, edge cases |
| `security-review` | Security engineer | OWASP Top 10, injection, auth, data exposure |
| `style-review` | Senior engineer | Naming, DRY, SRP, readability, best practices |
| `summary` | Lead engineer | Compiles all findings into a structured report with severity ratings |

**Default model:** `anthropic / claude-opus-4-6` → `openai / gpt-4o`

**Example:**
```bash
mvn exec:java -Dexec.args="--profile code-review" < src/UserService.kt
```

---

### `code-migration`

Four-step pipeline for migrating code between languages or frameworks. Prompts for `--from` and `--to` context.

| Step | Role | Task |
|---|---|---|
| `source-analysis` | Architect | Patterns, dependencies, migration challenges, complexity estimate |
| `migration-plan` | Migration expert | Dependency mapping, idiom transformations, step-by-step approach, test strategy |
| `code-generation` | Target-language developer | Full idiomatic migration preserving all business logic |
| `validation` | Dual-language expert | Logic parity, syntax correctness, idiomatic quality, confidence rating |

**Default model:** `anthropic / claude-opus-4-6` → `openai / gpt-4o`
**`code-generation` step override:** `anthropic / claude-sonnet-4-6` → `openai / gpt-4o-mini`

**Example:**
```bash
mvn exec:java -Dexec.args="--profile code-migration --from Java/Spring --to Kotlin/Ktor" < MyService.java
```

---

## Creating Custom Profiles

1. Create a new file in the `profiles/` directory, e.g., `profiles/my-pipeline.json`
2. Define steps — each step's `outputKey` becomes available as `{{step_<outputKey>}}` in later steps
3. Optionally add `contextPrompts` for user-supplied variables
4. Set `modelPreferences` with an ordered list for automatic fallback, or omit to use auto-routing

```json
{
  "name": "my-pipeline",
  "description": "My custom multi-step pipeline",
  "contextPrompts": [],
  "steps": [
    {
      "id": "draft",
      "name": "Draft",
      "systemPrompt": "You are a technical writer.",
      "userPromptTemplate": "Write a first draft about:\n{{input}}",
      "outputKey": "draft"
    },
    {
      "id": "refine",
      "name": "Refine",
      "systemPrompt": "You are an editor. Improve clarity and conciseness.",
      "userPromptTemplate": "Refine this draft:\n{{step_draft}}",
      "outputKey": "refined"
    }
  ],
  "modelPreferences": {
    "default": [
      { "provider": "anthropic", "model": "claude-sonnet-4-6" },
      { "provider": "openai",    "model": "gpt-4o" },
      { "provider": "ollama",    "model": "llama3.2" }
    ],
    "stepOverrides": {}
  }
}
```

Run it:
```
/profile my-pipeline
```

---

## Project Structure

```
AgenticAISandbox/
├── pom.xml
├── .env.example
├── .gitignore
├── profiles/
│   ├── code-review.json
│   ├── code-migration.json
│   └── github-pr-review.json
└── src/
    └── main/
        └── kotlin/
            ├── Main.kt
            ├── McpMain.kt
            ├── config/
            │   └── Config.kt
            ├── ai/
            │   ├── AIProvider.kt
            │   ├── ProviderRegistry.kt
            │   ├── providers/
            │   │   ├── OpenAIProvider.kt
            │   │   ├── AnthropicProvider.kt
            │   │   └── OllamaProvider.kt
            │   ├── router/
            │   │   ├── TaskType.kt
            │   │   ├── RouteTarget.kt
            │   │   └── ModelRouter.kt
            │   └── orchestration/
            │       ├── ProfileConfig.kt
            │       ├── OrchestrationResult.kt
            │       ├── ProfileLoader.kt
            │       └── OrchestrationEngine.kt
            ├── cli/
            │   ├── InteractiveCLI.kt
            │   └── SingleShotCLI.kt
            ├── github/
            │   ├── GitHubModels.kt
            │   ├── GitHubClient.kt
            │   └── GitHubReviewOrchestrator.kt
            └── mcp/
                ├── McpServer.kt
                └── GitHubMcpServer.kt
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `kotlin-stdlib` | 2.0.21 | Kotlin standard library |
| `okhttp` | 4.12.0 | HTTP client for all provider API calls |
| `gson` | 2.10.1 | JSON serialization / profile deserialization |
| `dotenv-kotlin` | 6.4.1 | `.env` file loading with env var precedence |
| `kotlin-test-junit5` | 2.0.21 | Test framework |
| `junit-jupiter` | 5.10.0 | JUnit 5 test runner |

**Build:**
```bash
mvn compile          # compile only
mvn test             # compile + run tests
mvn package          # produce JAR in target/
```
