 Local LLM Deployment with Ollama + Multi-Provider Model Routing

## A Complete Guide for Kotlin & Python Integration

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Ollama Setup & Model Deployment](#2-ollama-setup--model-deployment)
3. [Model Routing Strategy](#3-model-routing-strategy)
4. [Python Implementation](#4-python-implementation)
5. [Kotlin Implementation](#5-kotlin-implementation)
6. [Provider Configuration Reference](#6-provider-configuration-reference)
7. [Advanced Patterns](#7-advanced-patterns)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Overview

The system uses a **model router** that dynamically selects the best provider for each request based on task complexity, cost, latency requirements, and availability.

```
                         ┌─────────────────────┐
     User Request ──────▶│    Model Router      │
                         │  (complexity/cost/   │
                         │   latency analysis)  │
                         └──────┬──┬──┬──┬──────┘
                                │  │  │  │
                 ┌──────────────┘  │  │  └──────────────┐
                 ▼                 ▼  ▼                  ▼
          ┌──────────┐   ┌────────────────┐   ┌──────────────┐
          │  Ollama   │   │  Azure OpenAI  │   │   Anthropic  │
          │  (Local)  │   │  (Enterprise)  │   │   (Claude)   │
          │           │   │                │   │              │
          │ llama3    │   │ gpt-4o         │   │ claude-sonnet│
          │ mistral   │   │ gpt-4o-mini    │   │ claude-opus  │
          │ codellama │   │                │   │              │
          └──────────┘   └────────────────┘   └──────────────┘
                                  │
                          ┌───────┴────────┐
                          │ OpenAI Direct  │
                          │ (3rd Party)    │
                          │ gpt-4o         │
                          └────────────────┘
```

**Routing tiers:**

| Tier | Provider | Use Case | Latency | Cost |
|------|----------|----------|---------|------|
| Local | Ollama | Simple tasks, prototyping, offline, privacy-sensitive | ~50-200ms | Free |
| Standard | Azure OpenAI / OpenAI | Medium complexity, production workloads | ~500ms-2s | $$ |
| Premium | Claude Opus / GPT-4o | Complex reasoning, analysis, code generation | ~1-5s | $$$ |

---

## 2. Ollama Setup & Model Deployment

### 2.1 Installation

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download).

**Docker:**
```bash
# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# With NVIDIA GPU
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 2.2 Pull Models

```bash
# General purpose
ollama pull llama3.1:8b          # Good balance of speed/quality
ollama pull llama3.1:70b         # Higher quality, needs ~40GB VRAM
ollama pull mistral:7b           # Fast, good for simple tasks

# Code-specific
ollama pull codellama:13b        # Code generation
ollama pull deepseek-coder-v2:16b # Strong coding model

# Small & fast (for routing tier "local-fast")
ollama pull phi3:mini             # 3.8B, very fast
ollama pull gemma2:2b             # Tiny, instant responses

# Embedding models
ollama pull nomic-embed-text      # For RAG pipelines
```

### 2.3 Verify Installation

```bash
# Check server is running
curl http://localhost:11434/api/tags

# Test a model
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello, how are you?",
  "stream": false
}'

# Chat endpoint (OpenAI-compatible)
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

### 2.4 Ollama Configuration

```bash
# Environment variables
export OLLAMA_HOST=0.0.0.0:11434      # Listen on all interfaces
export OLLAMA_NUM_PARALLEL=4           # Concurrent requests
export OLLAMA_MAX_LOADED_MODELS=2      # Models in memory
export OLLAMA_KEEP_ALIVE=5m            # Unload after idle

# GPU configuration
export OLLAMA_GPU_OVERHEAD=0           # Reserved VRAM (bytes)
export CUDA_VISIBLE_DEVICES=0,1        # Multi-GPU
```

### 2.5 Custom Modelfiles

Create specialized models with system prompts:

```dockerfile
# Modelfile.code-assistant
FROM codellama:13b
SYSTEM "You are a senior software engineer. Provide concise, production-ready code. Always include error handling."
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
```

```bash
ollama create code-assistant -f Modelfile.code-assistant
```

---

## 3. Model Routing Strategy

### 3.1 Routing Decision Matrix

```
┌─────────────────────────────────────────────────────────┐
│                   ROUTING LOGIC                         │
│                                                         │
│  1. Is the data privacy-sensitive?                      │
│     YES → Ollama (local only)                           │
│                                                         │
│  2. Estimate task complexity (token count, domain):     │
│     LOW  (simple Q&A, formatting) → Ollama              │
│     MED  (summarization, basic code) → Azure/OpenAI     │
│     HIGH (complex reasoning, analysis) → Claude/GPT-4o  │
│                                                         │
│  3. Is low latency critical?                            │
│     YES → Ollama (local) or cached response             │
│                                                         │
│  4. Is the primary provider available?                  │
│     NO  → Fallback chain: Local → Azure → OpenAI → Claude│
│                                                         │
│  5. Budget constraint?                                  │
│     YES → Prefer Ollama → Azure → OpenAI                │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Complexity Classifier (Heuristic)

The router classifies requests into complexity tiers using:

- **Token count** of the prompt (longer = likely more complex)
- **Keyword signals**: "analyze", "compare", "explain why", "debug", "refactor" → higher complexity
- **Domain detection**: code, legal, medical → may need premium models
- **Conversation history depth**: multi-turn reasoning → higher tier
- **Explicit user override**: allow users to pin a provider

---

## 4. Python Implementation

### 4.1 Project Structure

```
model_router/
├── config.yaml            # Provider & routing configuration
├── router.py              # Core routing logic
├── providers/
│   ├── __init__.py
│   ├── base.py            # Abstract provider interface
│   ├── ollama_provider.py
│   ├── azure_openai.py
│   ├── openai_provider.py
│   └── anthropic_provider.py
├── classifier.py          # Complexity classifier
└── main.py                # Usage example
```

### 4.2 Configuration (`config.yaml`)

```yaml
providers:
  ollama:
    enabled: true
    base_url: "http://localhost:11434"
    models:
      fast: "phi3:mini"
      default: "llama3.1:8b"
      code: "codellama:13b"
    timeout: 30
    priority: 1  # Highest priority (tried first)

  azure_openai:
    enabled: true
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    api_version: "2024-10-21"
    models:
      default: "gpt-4o-mini"     # deployment name
      premium: "gpt-4o"
    timeout: 60
    priority: 2

  openai:
    enabled: true
    api_key: "${OPENAI_API_KEY}"
    models:
      default: "gpt-4o-mini"
      premium: "gpt-4o"
    timeout: 60
    priority: 3

  anthropic:
    enabled: true
    api_key: "${ANTHROPIC_API_KEY}"
    models:
      default: "claude-sonnet-4-5-20250929"
      premium: "claude-opus-4-6"
    timeout: 120
    priority: 4

routing:
  default_strategy: "complexity"  # complexity | cost | latency | privacy
  fallback_enabled: true
  max_retries: 2
  complexity_thresholds:
    low: 0.3       # → Ollama
    medium: 0.7    # → Azure/OpenAI
    high: 1.0      # → Claude/GPT-4o
```

### 4.3 Base Provider (`providers/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    content: str
    model: str
    provider: str
    usage: dict | None = None
    latency_ms: float = 0


class BaseProvider(ABC):
    """Abstract interface all providers implement."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
```

### 4.4 Ollama Provider (`providers/ollama_provider.py`)

```python
import time
import httpx
from typing import AsyncIterator
from .base import BaseProvider, ChatMessage, ChatResponse


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        start = time.monotonic()
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.monotonic() - start) * 1000

        return ChatResponse(
            content=data["message"]["content"],
            model=model,
            provider="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            latency_ms=elapsed,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with self.client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as resp:
            import json
            async for line in resp.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content", ""):
                        yield content

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    # --- Ollama-specific helpers ---

    async def list_models(self) -> list[str]:
        resp = await self.client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    async def pull_model(self, model: str) -> None:
        async with self.client.stream(
            "POST", f"{self.base_url}/api/pull", json={"name": model}
        ) as resp:
            async for line in resp.aiter_lines():
                pass  # Wait for download to finish
```

### 4.5 Azure OpenAI Provider (`providers/azure_openai.py`)

```python
import time
import httpx
from typing import AsyncIterator
from .base import BaseProvider, ChatMessage, ChatResponse


class AzureOpenAIProvider(BaseProvider):
    def __init__(self, endpoint: str, api_key: str, api_version: str = "2024-10-21", timeout: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
        )

    def _url(self, deployment: str) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        start = time.monotonic()
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self.client.post(self._url(model), json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.monotonic() - start) * 1000

        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="azure_openai",
            usage=data.get("usage"),
            latency_ms=elapsed,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with self.client.stream("POST", self._url(model), json=payload) as resp:
            import json
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content", ""):
                        yield content

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get(
                f"{self.endpoint}/openai/models?api-version={self.api_version}"
            )
            return resp.status_code == 200
        except Exception:
            return False
```

### 4.6 OpenAI Provider (`providers/openai_provider.py`)

```python
import time
import httpx
from typing import AsyncIterator
from .base import BaseProvider, ChatMessage, ChatResponse


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, timeout: int = 60):
        self.client = httpx.AsyncClient(
            timeout=timeout,
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        start = time.monotonic()
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.monotonic() - start) * 1000

        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="openai",
            usage=data.get("usage"),
            latency_ms=elapsed,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
            import json
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content", ""):
                        yield content

    async def health_check(self) -> bool:
        try:
            resp = await self.client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False
```

### 4.7 Anthropic Provider (`providers/anthropic_provider.py`)

```python
import time
import httpx
from typing import AsyncIterator
from .base import BaseProvider, ChatMessage, ChatResponse


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, timeout: int = 120):
        self.client = httpx.AsyncClient(
            timeout=timeout,
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        start = time.monotonic()

        # Extract system message if present
        system = None
        chat_msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        payload = {
            "model": model,
            "messages": chat_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        resp = await self.client.post("/v1/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.monotonic() - start) * 1000

        content = "".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )
        return ChatResponse(
            content=content,
            model=model,
            provider="anthropic",
            usage=data.get("usage"),
            latency_ms=elapsed,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        system = None
        chat_msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_msgs.append({"role": m.role, "content": m.content})

        payload = {
            "model": model,
            "messages": chat_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self.client.stream("POST", "/v1/messages", json=payload) as resp:
            import json
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "content_block_delta":
                        if text := event["delta"].get("text", ""):
                            yield text

    async def health_check(self) -> bool:
        try:
            # Anthropic doesn't have a health endpoint; try a minimal call
            resp = await self.client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return resp.status_code == 200
        except Exception:
            return False
```

### 4.8 Complexity Classifier (`classifier.py`)

```python
import re

# Keywords that suggest higher complexity
HIGH_COMPLEXITY_SIGNALS = {
    "analyze", "compare", "contrast", "evaluate", "synthesize",
    "debug", "refactor", "architect", "design pattern", "optimize",
    "explain why", "trade-off", "pros and cons", "deep dive",
    "research", "comprehensive", "detailed analysis",
}
MEDIUM_COMPLEXITY_SIGNALS = {
    "summarize", "write", "generate", "create", "convert",
    "translate", "implement", "function", "class", "api",
}
CODE_SIGNALS = {
    "code", "function", "class", "bug", "error", "stack trace",
    "implement", "refactor", "test", "debug", "deploy",
}


def classify_complexity(prompt: str, history_turns: int = 0) -> float:
    """
    Returns a complexity score between 0.0 and 1.0.
    
    0.0 - 0.3: Low    → Ollama (local)
    0.3 - 0.7: Medium → Azure OpenAI / OpenAI
    0.7 - 1.0: High   → Claude / GPT-4o
    """
    score = 0.0
    prompt_lower = prompt.lower()
    words = prompt_lower.split()
    word_count = len(words)

    # Length-based scoring
    if word_count > 500:
        score += 0.3
    elif word_count > 100:
        score += 0.15
    elif word_count < 20:
        score -= 0.1

    # Keyword signals
    for signal in HIGH_COMPLEXITY_SIGNALS:
        if signal in prompt_lower:
            score += 0.25
            break  # Don't over-count

    for signal in MEDIUM_COMPLEXITY_SIGNALS:
        if signal in prompt_lower:
            score += 0.1
            break

    # Code detection
    has_code = bool(re.search(r"```[\s\S]+```", prompt)) or any(
        s in prompt_lower for s in CODE_SIGNALS
    )
    if has_code:
        score += 0.15

    # Multi-turn bonus
    if history_turns > 5:
        score += 0.15
    elif history_turns > 2:
        score += 0.05

    # Question complexity
    question_marks = prompt.count("?")
    if question_marks > 3:
        score += 0.1

    return max(0.0, min(1.0, score))


def detect_privacy_requirement(prompt: str) -> bool:
    """Check if the prompt contains data that should stay local."""
    privacy_signals = [
        r"\b(ssn|social security)\b",
        r"\b(password|secret|token|api.key)\b",
        r"\b(credit card|bank account)\b",
        r"\b(medical|diagnosis|patient)\b",
        r"\b(confidential|internal only|proprietary)\b",
    ]
    prompt_lower = prompt.lower()
    return any(re.search(p, prompt_lower) for p in privacy_signals)
```

### 4.9 Model Router (`router.py`)

```python
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from providers.base import BaseProvider, ChatMessage, ChatResponse
from classifier import classify_complexity, detect_privacy_requirement

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    COMPLEXITY = "complexity"
    COST = "cost"
    LATENCY = "latency"
    PRIVACY = "privacy"


@dataclass
class ProviderConfig:
    name: str
    provider: BaseProvider
    priority: int
    models: dict[str, str]  # tier -> model name
    enabled: bool = True


class ModelRouter:
    """Routes requests to the optimal provider based on strategy."""

    def __init__(
        self,
        providers: list[ProviderConfig],
        strategy: RoutingStrategy = RoutingStrategy.COMPLEXITY,
        fallback_enabled: bool = True,
        max_retries: int = 2,
    ):
        self.providers = sorted(providers, key=lambda p: p.priority)
        self.strategy = strategy
        self.fallback_enabled = fallback_enabled
        self.max_retries = max_retries
        self._health_cache: dict[str, bool] = {}

    async def route(
        self,
        messages: list[ChatMessage],
        strategy: RoutingStrategy | None = None,
        force_provider: str | None = None,
        force_model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        """Route a chat request to the best provider."""
        active_strategy = strategy or self.strategy

        # Force a specific provider if requested
        if force_provider:
            return await self._call_provider(
                force_provider, messages, force_model, temperature, max_tokens
            )

        # Get the prompt for classification
        prompt = messages[-1].content if messages else ""
        history_turns = len([m for m in messages if m.role == "user"])

        # Privacy check always overrides
        if detect_privacy_requirement(prompt):
            logger.info("Privacy-sensitive content detected → routing to Ollama")
            return await self._call_with_fallback(
                ["ollama"], messages, "default", temperature, max_tokens
            )

        # Determine provider order based on strategy
        provider_order = self._select_providers(
            active_strategy, prompt, history_turns
        )

        return await self._call_with_fallback(
            provider_order, messages, None, temperature, max_tokens,
            prompt=prompt, history_turns=history_turns,
        )

    def _select_providers(
        self, strategy: RoutingStrategy, prompt: str, history_turns: int
    ) -> list[str]:
        """Select ordered provider list based on strategy."""

        if strategy == RoutingStrategy.LATENCY:
            # Local first, then by priority
            return [p.name for p in self.providers if p.enabled]

        if strategy == RoutingStrategy.COST:
            # Free (local) first, then cheapest cloud
            return ["ollama", "azure_openai", "openai", "anthropic"]

        if strategy == RoutingStrategy.PRIVACY:
            return ["ollama"]

        # COMPLEXITY strategy (default)
        complexity = classify_complexity(prompt, history_turns)
        logger.info(f"Complexity score: {complexity:.2f}")

        if complexity < 0.3:
            return ["ollama", "azure_openai", "openai"]
        elif complexity < 0.7:
            return ["azure_openai", "openai", "ollama", "anthropic"]
        else:
            return ["anthropic", "openai", "azure_openai", "ollama"]

    def _get_model_for_provider(
        self, provider_name: str, prompt: str, history_turns: int
    ) -> str:
        """Select the right model tier for a provider."""
        complexity = classify_complexity(prompt, history_turns)
        provider = next(p for p in self.providers if p.name == provider_name)

        if complexity >= 0.7 and "premium" in provider.models:
            return provider.models["premium"]
        elif "code" in provider.models and self._is_code_task(prompt):
            return provider.models["code"]
        elif complexity < 0.2 and "fast" in provider.models:
            return provider.models["fast"]
        return provider.models.get("default", list(provider.models.values())[0])

    @staticmethod
    def _is_code_task(prompt: str) -> bool:
        code_indicators = ["```", "function", "class ", "def ", "import ", "bug", "error"]
        return any(ind in prompt.lower() for ind in code_indicators)

    async def _call_with_fallback(
        self,
        provider_order: list[str],
        messages: list[ChatMessage],
        model_override: str | None,
        temperature: float,
        max_tokens: int,
        prompt: str = "",
        history_turns: int = 0,
    ) -> ChatResponse:
        """Try providers in order with fallback."""
        last_error = None

        for provider_name in provider_order:
            pconfig = next(
                (p for p in self.providers if p.name == provider_name and p.enabled),
                None,
            )
            if not pconfig:
                continue

            model = model_override or self._get_model_for_provider(
                provider_name, prompt, history_turns
            )

            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"Trying {provider_name}/{model} (attempt {attempt + 1})"
                    )
                    response = await pconfig.provider.chat(
                        messages, model, temperature, max_tokens
                    )
                    logger.info(
                        f"Success: {provider_name}/{model} "
                        f"({response.latency_ms:.0f}ms)"
                    )
                    return response
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{provider_name}/{model} attempt {attempt + 1} failed: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))

            if not self.fallback_enabled:
                break

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )

    async def _call_provider(
        self,
        provider_name: str,
        messages: list[ChatMessage],
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> ChatResponse:
        pconfig = next(p for p in self.providers if p.name == provider_name)
        model = model or pconfig.models.get("default")
        return await pconfig.provider.chat(messages, model, temperature, max_tokens)

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all providers."""
        results = {}
        for p in self.providers:
            results[p.name] = await p.provider.health_check()
        self._health_cache = results
        return results
```

### 4.10 Usage Example (`main.py`)

```python
import asyncio
import os
from providers.ollama_provider import OllamaProvider
from providers.azure_openai import AzureOpenAIProvider
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.base import ChatMessage
from router import ModelRouter, ProviderConfig, RoutingStrategy


async def main():
    # Initialize providers
    providers = [
        ProviderConfig(
            name="ollama",
            provider=OllamaProvider(),
            priority=1,
            models={"fast": "phi3:mini", "default": "llama3.1:8b", "code": "codellama:13b"},
        ),
        ProviderConfig(
            name="azure_openai",
            provider=AzureOpenAIProvider(
                endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
            ),
            priority=2,
            models={"default": "gpt-4o-mini", "premium": "gpt-4o"},
        ),
        ProviderConfig(
            name="openai",
            provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
            priority=3,
            models={"default": "gpt-4o-mini", "premium": "gpt-4o"},
        ),
        ProviderConfig(
            name="anthropic",
            provider=AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
            priority=4,
            models={"default": "claude-sonnet-4-5-20250929", "premium": "claude-opus-4-6"},
        ),
    ]

    router = ModelRouter(providers, strategy=RoutingStrategy.COMPLEXITY)

    # --- Example 1: Simple question → routed to Ollama ---
    response = await router.route([
        ChatMessage(role="user", content="What is 2 + 2?")
    ])
    print(f"[{response.provider}/{response.model}] {response.content}")

    # --- Example 2: Complex analysis → routed to Claude/GPT-4o ---
    response = await router.route([
        ChatMessage(role="user",
                    content="Analyze the trade-offs between microservices and monolithic "
                            "architecture for a startup with 5 engineers building a fintech product.")
    ])
    print(f"[{response.provider}/{response.model}] {response.content[:200]}...")

    # --- Example 3: Privacy-sensitive → forced local ---
    response = await router.route([
        ChatMessage(role="user",
                    content="Review this patient medical record and extract key diagnoses...")
    ])
    print(f"[{response.provider}/{response.model}] Routed to local (privacy)")

    # --- Example 4: Force a specific provider ---
    response = await router.route(
        [ChatMessage(role="user", content="Hello!")],
        force_provider="anthropic",
    )
    print(f"[{response.provider}/{response.model}] {response.content}")

    # --- Example 5: Cost-optimized strategy ---
    response = await router.route(
        [ChatMessage(role="user", content="Summarize this article...")],
        strategy=RoutingStrategy.COST,
    )
    print(f"[{response.provider}/{response.model}] Cost-optimized routing")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 5. Kotlin Implementation

### 5.1 Project Setup (Gradle `build.gradle.kts`)

```kotlin
plugins {
    kotlin("jvm") version "2.0.21"
    kotlin("plugin.serialization") version "2.0.21"
}

dependencies {
    // HTTP client
    implementation("io.ktor:ktor-client-core:3.0.3")
    implementation("io.ktor:ktor-client-cio:3.0.3")
    implementation("io.ktor:ktor-client-content-negotiation:3.0.3")
    implementation("io.ktor:ktor-serialization-kotlinx-json:3.0.3")

    // Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")

    // Logging
    implementation("io.github.microutils:kotlin-logging-jvm:3.0.5")
    implementation("ch.qos.logback:logback-classic:1.5.12")

    // Config
    implementation("com.sksamuel.hoplite:hoplite-core:2.8.2")
    implementation("com.sksamuel.hoplite:hoplite-yaml:2.8.2")
}
```

### 5.2 Data Models

```kotlin
// models/ChatModels.kt
package models

import kotlinx.serialization.Serializable

@Serializable
data class ChatMessage(
    val role: String, // "system", "user", "assistant"
    val content: String
)

data class ChatResponse(
    val content: String,
    val model: String,
    val provider: String,
    val usage: Map? = null,
    val latencyMs: Long = 0
)

enum class RoutingStrategy {
    COMPLEXITY, COST, LATENCY, PRIVACY
}

data class ProviderConfig(
    val name: String,
    val provider: LLMProvider,
    val priority: Int,
    val models: Map, // tier -> model name
    val enabled: Boolean = true
)
```

### 5.3 Provider Interface

```kotlin
// providers/LLMProvider.kt
package providers

import kotlinx.coroutines.flow.Flow
import models.ChatMessage
import models.ChatResponse

interface LLMProvider {
    suspend fun chat(
        messages: List,
        model: String? = null,
        temperature: Double = 0.7,
        maxTokens: Int = 2048
    ): ChatResponse

    fun chatStream(
        messages: List,
        model: String? = null,
        temperature: Double = 0.7,
        maxTokens: Int = 2048
    ): Flow

    suspend fun healthCheck(): Boolean
}
```

### 5.4 Ollama Provider (Kotlin)

```kotlin
// providers/OllamaProvider.kt
package providers

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.*
import models.ChatMessage
import models.ChatResponse

class OllamaProvider(
    private val baseUrl: String = "http://localhost:11434",
    timeoutMs: Long = 30_000
) : LLMProvider {

    private val client = HttpClient(CIO) {
        install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
        engine { requestTimeout = timeoutMs }
    }
    private val json = Json { ignoreUnknownKeys = true }

    override suspend fun chat(
        messages: List,
        model: String?,
        temperature: Double,
        maxTokens: Int
    ): ChatResponse {
        val start = System.currentTimeMillis()
        val effectiveModel = model ?: "llama3.1:8b"

        val payload = buildJsonObject {
            put("model", effectiveModel)
            put("stream", false)
            putJsonArray("messages") {
                messages.forEach { msg ->
                    addJsonObject {
                        put("role", msg.role)
                        put("content", msg.content)
                    }
                }
            }
            putJsonObject("options") {
                put("temperature", temperature)
                put("num_predict", maxTokens)
            }
        }

        val response = client.post("$baseUrl/api/chat") {
            contentType(ContentType.Application.Json)
            setBody(payload.toString())
        }
        val data = json.parseToJsonElement(response.body()).jsonObject
        val elapsed = System.currentTimeMillis() - start

        return ChatResponse(
            content = data["message"]!!.jsonObject["content"]!!.jsonPrimitive.content,
            model = effectiveModel,
            provider = "ollama",
            usage = mapOf(
                "prompt_tokens" to (data["prompt_eval_count"]?.jsonPrimitive?.int ?: 0),
                "completion_tokens" to (data["eval_count"]?.jsonPrimitive?.int ?: 0)
            ),
            latencyMs = elapsed
        )
    }

    override fun chatStream(
        messages: List,
        model: String?,
        temperature: Double,
        maxTokens: Int
    ): Flow = flow {
        val effectiveModel = model ?: "llama3.1:8b"
        val payload = buildJsonObject {
            put("model", effectiveModel)
            put("stream", true)
            putJsonArray("messages") {
                messages.forEach { msg ->
                    addJsonObject {
                        put("role", msg.role)
                        put("content", msg.content)
                    }
                }
            }
            putJsonObject("options") {
                put("temperature", temperature)
                put("num_predict", maxTokens)
            }
        }

        client.preparePost("$baseUrl/api/chat") {
            contentType(ContentType.Application.Json)
            setBody(payload.toString())
        }.execute { response ->
            val channel = response.bodyAsChannel()
            val buffer = StringBuilder()
            while (!channel.isClosedForRead) {
                val byte = channel.readByte()
                val char = byte.toInt().toChar()
                buffer.append(char)
                if (char == '\n') {
                    val line = buffer.toString().trim()
                    if (line.isNotEmpty()) {
                        val chunk = json.parseToJsonElement(line).jsonObject
                        val content = chunk["message"]?.jsonObject?.get("content")
                            ?.jsonPrimitive?.content ?: ""
                        if (content.isNotEmpty()) emit(content)
                    }
                    buffer.clear()
                }
            }
        }
    }

    override suspend fun healthCheck(): Boolean = try {
        client.get("$baseUrl/api/tags").status == HttpStatusCode.OK
    } catch (e: Exception) {
        false
    }

    suspend fun listModels(): List {
        val response = client.get("$baseUrl/api/tags")
        val data = json.parseToJsonElement(response.body()).jsonObject
        return data["models"]?.jsonArray?.map {
            it.jsonObject["name"]!!.jsonPrimitive.content
        } ?: emptyList()
    }
}
```

### 5.5 Azure OpenAI Provider (Kotlin)

```kotlin
// providers/AzureOpenAIProvider.kt
package providers

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.request.*
import io.ktor.http.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.serialization.json.*
import models.ChatMessage
import models.ChatResponse

class AzureOpenAIProvider(
    private val endpoint: String,
    private val apiKey: String,
    private val apiVersion: String = "2024-10-21",
    timeoutMs: Long = 60_000
) : LLMProvider {

    private val client = HttpClient(CIO) {
        engine { requestTimeout = timeoutMs }
    }
    private val json = Json { ignoreUnknownKeys = true }

    private fun url(deployment: String) =
        "$endpoint/openai/deployments/$deployment/chat/completions?api-version=$apiVersion"

    override suspend fun chat(
        messages: List,
        model: String?,
        temperature: Double,
        maxTokens: Int
    ): ChatResponse {
        val start = System.currentTimeMillis()
        val deployment = model ?: "gpt-4o-mini"

        val payload = buildJsonObject {
            putJsonArray("messages") {
                messages.forEach { msg ->
                    addJsonObject {
                        put("role", msg.role)
                        put("content", msg.content)
                    }
                }
            }
            put("temperature", temperature)
            put("max_tokens", maxTokens)
        }

        val response = client.post(url(deployment)) {
            header("api-key", apiKey)
            contentType(ContentType.Application.Json)
            setBody(payload.toString())
        }
        val data = json.parseToJsonElement(response.body()).jsonObject
        val elapsed = System.currentTimeMillis() - start

        val content = data["choices"]!!.jsonArray[0]
            .jsonObject["message"]!!.jsonObject["content"]!!.jsonPrimitive.content

        return ChatResponse(
            content = content,
            model = deployment,
            provider = "azure_openai",
            latencyMs = elapsed
        )
    }

    override fun chatStream(
        messages: List,
        model: String?,
        temperature: Double,
        maxTokens: Int
    ): Flow = flow {
        // Similar SSE parsing as OpenAI — omitted for brevity
        // Use the same pattern as OllamaProvider.chatStream with SSE parsing
    }

    override suspend fun healthCheck(): Boolean = try {
        client.get("$endpoint/openai/models?api-version=$apiVersion") {
            header("api-key", apiKey)
        }.status == HttpStatusCode.OK
    } catch (e: Exception) {
        false
    }
}
```

### 5.6 Model Router (Kotlin)

```kotlin
// router/ModelRouter.kt
package router

import kotlinx.coroutines.delay
import models.*
import mu.KotlinLogging
import providers.LLMProvider

private val logger = KotlinLogging.logger {}

class ModelRouter(
    private val providers: List,
    private val defaultStrategy: RoutingStrategy = RoutingStrategy.COMPLEXITY,
    private val fallbackEnabled: Boolean = true,
    private val maxRetries: Int = 2
) {
    private val sortedProviders = providers.sortedBy { it.priority }

    suspend fun route(
        messages: List,
        strategy: RoutingStrategy? = null,
        forceProvider: String? = null,
        forceModel: String? = null,
        temperature: Double = 0.7,
        maxTokens: Int = 2048
    ): ChatResponse {
        val activeStrategy = strategy ?: defaultStrategy
        val prompt = messages.lastOrNull()?.content ?: ""
        val historyTurns = messages.count { it.role == "user" }

        // Force provider
        if (forceProvider != null) {
            val pc = sortedProviders.first { it.name == forceProvider }
            val model = forceModel ?: pc.models["default"]!!
            return pc.provider.chat(messages, model, temperature, maxTokens)
        }

        // Privacy override
        if (ComplexityClassifier.detectPrivacy(prompt)) {
            logger.info { "Privacy-sensitive → routing to Ollama" }
            return callWithFallback(listOf("ollama"), messages, prompt, historyTurns, temperature, maxTokens)
        }

        val providerOrder = selectProviders(activeStrategy, prompt, historyTurns)
        return callWithFallback(providerOrder, messages, prompt, historyTurns, temperature, maxTokens)
    }

    private fun selectProviders(
        strategy: RoutingStrategy,
        prompt: String,
        historyTurns: Int
    ): List {
        return when (strategy) {
            RoutingStrategy.LATENCY -> sortedProviders.filter { it.enabled }.map { it.name }
            RoutingStrategy.COST -> listOf("ollama", "azure_openai", "openai", "anthropic")
            RoutingStrategy.PRIVACY -> listOf("ollama")
            RoutingStrategy.COMPLEXITY -> {
                val complexity = ComplexityClassifier.classify(prompt, historyTurns)
                logger.info { "Complexity: ${"%.2f".format(complexity)}" }
                when {
                    complexity < 0.3 -> listOf("ollama", "azure_openai", "openai")
                    complexity < 0.7 -> listOf("azure_openai", "openai", "ollama", "anthropic")
                    else -> listOf("anthropic", "openai", "azure_openai", "ollama")
                }
            }
        }
    }

    private fun getModelForProvider(name: String, prompt: String, turns: Int): String {
        val pc = sortedProviders.first { it.name == name }
        val complexity = ComplexityClassifier.classify(prompt, turns)
        return when {
            complexity >= 0.7 && "premium" in pc.models -> pc.models["premium"]!!
            complexity < 0.2 && "fast" in pc.models -> pc.models["fast"]!!
            else -> pc.models["default"] ?: pc.models.values.first()
        }
    }

    private suspend fun callWithFallback(
        order: List,
        messages: List,
        prompt: String,
        historyTurns: Int,
        temperature: Double,
        maxTokens: Int
    ): ChatResponse {
        var lastError: Exception? = null

        for (providerName in order) {
            val pc = sortedProviders.firstOrNull { it.name == providerName && it.enabled }
                ?: continue
            val model = getModelForProvider(providerName, prompt, historyTurns)

            repeat(maxRetries) { attempt ->
                try {
                    logger.info { "Trying $providerName/$model (attempt ${attempt + 1})" }
                    val response = pc.provider.chat(messages, model, temperature, maxTokens)
                    logger.info { "✓ $providerName/$model (${response.latencyMs}ms)" }
                    return response
                } catch (e: Exception) {
                    lastError = e
                    logger.warn { "$providerName/$model failed: ${e.message}" }
                    if (attempt < maxRetries - 1) delay(500L * (attempt + 1))
                }
            }
            if (!fallbackEnabled) break
        }
        throw RuntimeException("All providers failed: ${lastError?.message}", lastError)
    }
}
```

### 5.7 Complexity Classifier (Kotlin)

```kotlin
// router/ComplexityClassifier.kt
package router

object ComplexityClassifier {

    private val HIGH_SIGNALS = setOf(
        "analyze", "compare", "contrast", "evaluate", "synthesize",
        "debug", "refactor", "architect", "optimize", "trade-off",
        "pros and cons", "deep dive", "comprehensive"
    )
    private val MEDIUM_SIGNALS = setOf(
        "summarize", "write", "generate", "create", "convert",
        "translate", "implement", "function", "class", "api"
    )
    private val PRIVACY_PATTERNS = listOf(
        Regex("\\b(ssn|social security)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(password|secret|token|api.key)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(credit card|bank account)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(medical|diagnosis|patient)\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(confidential|internal only|proprietary)\\b", RegexOption.IGNORE_CASE),
    )

    fun classify(prompt: String, historyTurns: Int = 0): Double {
        var score = 0.0
        val lower = prompt.lowercase()
        val wordCount = lower.split("\\s+".toRegex()).size

        // Length
        score += when {
            wordCount > 500 -> 0.3
            wordCount > 100 -> 0.15
            wordCount < 20 -> -0.1
            else -> 0.0
        }

        // Keywords
        if (HIGH_SIGNALS.any { it in lower }) score += 0.25
        if (MEDIUM_SIGNALS.any { it in lower }) score += 0.1

        // Code
        if ("```" in prompt || listOf("function", "class ", "def ", "bug", "error").any { it in lower })
            score += 0.15

        // History depth
        score += when {
            historyTurns > 5 -> 0.15
            historyTurns > 2 -> 0.05
            else -> 0.0
        }

        return score.coerceIn(0.0, 1.0)
    }

    fun detectPrivacy(prompt: String): Boolean =
        PRIVACY_PATTERNS.any { it.containsMatchIn(prompt) }
}
```

### 5.8 Kotlin Usage

```kotlin
// Main.kt
import kotlinx.coroutines.runBlocking
import models.*
import providers.*
import router.ModelRouter

fun main() = runBlocking {
    val providers = listOf(
        ProviderConfig(
            name = "ollama",
            provider = OllamaProvider(),
            priority = 1,
            models = mapOf("fast" to "phi3:mini", "default" to "llama3.1:8b", "code" to "codellama:13b")
        ),
        ProviderConfig(
            name = "azure_openai",
            provider = AzureOpenAIProvider(
                endpoint = System.getenv("AZURE_OPENAI_ENDPOINT"),
                apiKey = System.getenv("AZURE_OPENAI_API_KEY")
            ),
            priority = 2,
            models = mapOf("default" to "gpt-4o-mini", "premium" to "gpt-4o")
        ),
        ProviderConfig(
            name = "openai",
            provider = OpenAIProvider(apiKey = System.getenv("OPENAI_API_KEY")),
            priority = 3,
            models = mapOf("default" to "gpt-4o-mini", "premium" to "gpt-4o")
        ),
        ProviderConfig(
            name = "anthropic",
            provider = AnthropicProvider(apiKey = System.getenv("ANTHROPIC_API_KEY")),
            priority = 4,
            models = mapOf("default" to "claude-sonnet-4-5-20250929", "premium" to "claude-opus-4-6")
        )
    )

    val router = ModelRouter(providers)

    // Simple question → Ollama
    val r1 = router.route(listOf(ChatMessage("user", "What is 2+2?")))
    println("[${r1.provider}/${r1.model}] ${r1.content}")

    // Complex analysis → Claude/GPT-4o
    val r2 = router.route(listOf(
        ChatMessage("user", "Analyze trade-offs between event sourcing and CRUD for a fintech platform")
    ))
    println("[${r2.provider}/${r2.model}] ${r2.content.take(200)}...")

    // Force provider
    val r3 = router.route(
        messages = listOf(ChatMessage("user", "Hello")),
        forceProvider = "anthropic"
    )
    println("[${r3.provider}/${r3.model}] ${r3.content}")
}
```

---

## 6. Provider Configuration Reference

### 6.1 Environment Variables

```bash
# .env file
# Ollama (no auth needed for local)
OLLAMA_HOST=http://localhost:11434

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key

# OpenAI (direct)
OPENAI_API_KEY=sk-your-openai-key

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### 6.2 Model Comparison (for routing decisions)

| Model | Provider | Speed | Quality | Cost/1M tokens | Best For |
|-------|----------|-------|---------|-----------------|----------|
| phi3:mini | Ollama | ★★★★★ | ★★ | Free | Simple Q&A, classification |
| llama3.1:8b | Ollama | ★★★★ | ★★★ | Free | General tasks, offline |
| codellama:13b | Ollama | ★★★ | ★★★ | Free | Code generation (local) |
| gpt-4o-mini | Azure/OpenAI | ★★★★ | ★★★★ | ~$0.30 | Balanced cost/quality |
| gpt-4o | Azure/OpenAI | ★★★ | ★★★★★ | ~$5.00 | Complex reasoning |
| claude-sonnet-4-5 | Anthropic | ★★★ | ★★★★★ | ~$3.00 | Analysis, writing, code |
| claude-opus-4-6 | Anthropic | ★★ | ★★★★★ | ~$15.00 | Most complex tasks |

### 6.3 API Endpoint Summary

| Provider | Base URL | Auth Header |
|----------|----------|-------------|
| Ollama | `http://localhost:11434/api/chat` | None |
| Ollama (OpenAI compat) | `http://localhost:11434/v1/chat/completions` | None |
| Azure OpenAI | `{endpoint}/openai/deployments/{model}/chat/completions?api-version=...` | `api-key: {key}` |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `Authorization: Bearer {key}` |
| Anthropic | `https://api.anthropic.com/v1/messages` | `x-api-key: {key}` |

---

## 7. Advanced Patterns

### 7.1 Semantic Caching

Cache responses for similar prompts to reduce cost and latency:

```python
import hashlib
from functools import lru_cache

class SemanticCache:
    """Simple cache using prompt hashing. 
    For production, use embedding similarity with a vector DB."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, ChatResponse] = {}
        self._max_size = max_size

    def _hash(self, messages: list[ChatMessage]) -> str:
        content = "|".join(f"{m.role}:{m.content}" for m in messages)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, messages: list[ChatMessage]) -> ChatResponse | None:
        return self._cache.get(self._hash(messages))

    def put(self, messages: list[ChatMessage], response: ChatResponse):
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[self._hash(messages)] = response
```

### 7.2 Circuit Breaker Pattern

```python
import time
from dataclasses import dataclass

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    _failures: int = 0
    _last_failure: float = 0.0
    _state: str = "closed"  # closed, open, half-open

    def record_failure(self):
        self._failures += 1
        self._last_failure = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = "open"

    def record_success(self):
        self._failures = 0
        self._state = "closed"

    def can_execute(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._last_failure > self.recovery_timeout:
                self._state = "half-open"
                return True
            return False
        return True  # half-open: allow one attempt
```

### 7.3 Load Balancing Across Multiple Ollama Instances

```python
class OllamaLoadBalancer:
    """Round-robin across multiple Ollama servers."""
    
    def __init__(self, urls: list[str]):
        self.providers = [OllamaProvider(url) for url in urls]
        self._index = 0

    async def get_healthy_provider(self) -> OllamaProvider:
        for _ in range(len(self.providers)):
            provider = self.providers[self._index % len(self.providers)]
            self._index += 1
            if await provider.health_check():
                return provider
        raise RuntimeError("No healthy Ollama instances")
```

### 7.4 Observability & Logging

```python
# Add to router for production monitoring
import json, time

class RoutingMetrics:
    def __init__(self):
        self.requests = []

    def record(self, provider: str, model: str, latency_ms: float, 
               complexity: float, success: bool):
        self.requests.append({
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "complexity": complexity,
            "success": success,
        })

    def summary(self) -> dict:
        by_provider = {}
        for r in self.requests:
            p = r["provider"]
            if p not in by_provider:
                by_provider[p] = {"count": 0, "failures": 0, "total_latency": 0}
            by_provider[p]["count"] += 1
            by_provider[p]["total_latency"] += r["latency_ms"]
            if not r["success"]:
                by_provider[p]["failures"] += 1
        return by_provider
```

---

## 8. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Ollama not responding | Check `systemctl status ollama` or `ollama serve` is running |
| Model not found | Run `ollama pull model-name` first |
| Out of VRAM | Use smaller quantization (`llama3.1:8b-q4_0`) or set `OLLAMA_MAX_LOADED_MODELS=1` |
| Azure 401 errors | Verify `api-key` header and deployment name matches exactly |
| OpenAI rate limits | Implement exponential backoff; the router's retry logic handles this |
| Anthropic 529 (overloaded) | Fallback to OpenAI; increase `max_retries` |
| Slow first Ollama response | First request loads model into VRAM; set `OLLAMA_KEEP_ALIVE=30m` |
| Streaming not working | Ensure `stream: true` in payload and handle SSE/NDJSON correctly |

### Health Check Script

```bash
#!/bin/bash
echo "=== Provider Health Check ==="
echo -n "Ollama:       "; curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ OK" || echo "✗ DOWN"
echo -n "Azure OpenAI: "; curl -sf -H "api-key: $AZURE_OPENAI_API_KEY" \
  "$AZURE_OPENAI_ENDPOINT/openai/models?api-version=2024-10-21" > /dev/null && echo "✓ OK" || echo "✗ DOWN"
echo -n "OpenAI:       "; curl -sf -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models > /dev/null && echo "✓ OK" || echo "✗ DOWN"
echo -n "Anthropic:    "; curl -sf -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" -d '{"model":"claude-sonnet-4-5-20250929","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
  https://api.anthropic.com/v1/messages > /dev/null && echo "✓ OK" || echo "✗ DOWN"
```

---

## Quick Start Checklist

1. **Install Ollama** and pull at least one model (`ollama pull llama3.1:8b`)
2. **Set environment variables** for cloud providers (`AZURE_OPENAI_ENDPOINT`, API keys)
3. **Install dependencies**: `pip install httpx pyyaml` (Python) or add Ktor to Gradle (Kotlin)
4. **Copy the provider implementations** and router from this guide
5. **Configure `config.yaml`** with your endpoints and model deployments
6. **Run health checks** to verify all providers are accessible
7. **Start routing** — the complexity classifier handles provider selection automatically
