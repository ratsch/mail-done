# LLM Configuration Guide

This guide covers configuring LLM providers for email classification, embeddings, and other AI features.

## Related Documentation

- [Deployment Guide](DEPLOYMENT.md) - Full deployment instructions
- [Email Processing](PROCESS_INBOX.md) - Using AI classification

---

## Quick Start

For basic setup, you only need one provider. Add to `.env`:

```bash
# Option A: OpenAI (simplest)
OPENAI_API_KEY=sk-...

# Option B: Azure OpenAI
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

---

## Supported Providers

| Provider | Use Case | Configuration |
|----------|----------|---------------|
| OpenAI | Simple setup, latest models | `OPENAI_API_KEY` |
| Azure OpenAI | Enterprise, regional compliance | Endpoint + API key |
| Anthropic | Claude models | `ANTHROPIC_API_KEY` |
| Ollama | Local/self-hosted | Base URL |

---

## Provider Configuration

### OpenAI

```bash
# .env
OPENAI_API_KEY=sk-proj-...

# For embeddings
EMBEDDING_PROVIDER=openai
```

### Azure OpenAI

```bash
# .env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# For embeddings
EMBEDDING_PROVIDER=azure
```

**Finding your Azure endpoint:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Keys and Endpoint → Copy the endpoint URL

### Anthropic (Claude)

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### Ollama (Local)

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

## Multi-Region Azure Setup

For high availability or to use models deployed in different regions, configure multiple Azure endpoints.

### Why Multi-Region?

- **Model availability**: Some models are only deployed in specific regions
- **Latency**: Route to the nearest region
- **Quota management**: Spread load across regions
- **Compliance**: Keep data in specific regions

### Configuration

**Step 1: Add endpoints to `.env`**

```bash
# Primary region (West Europe)
AZURE_OPENAI_API_KEY=your-west-europe-key
AZURE_OPENAI_ENDPOINT=https://your-resource-westeurope.openai.azure.com

# Secondary region (Sweden)
AZURE_OPENAI_SWEDEN_API_KEY=your-sweden-key
AZURE_OPENAI_SWEDEN_ENDPOINT=https://your-resource-sweden.openai.azure.com

# API version (same for all)
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

**Step 2: Configure model routing in `llm_endpoints.yaml`**

Create or edit `backend/core/ai/config/llm_endpoints.yaml`:

```yaml
# Provider profiles (define credentials once)
providers:
  azure_west_europe:
    type: azure
    endpoint_env: AZURE_OPENAI_ENDPOINT
    api_key_env: AZURE_OPENAI_API_KEY
    api_version: "2025-04-01-preview"

  azure_sweden:
    type: azure
    endpoint_env: AZURE_OPENAI_SWEDEN_ENDPOINT
    api_key_env: AZURE_OPENAI_SWEDEN_API_KEY
    api_version: "2025-04-01-preview"

  openai:
    type: openai
    api_key_env: OPENAI_API_KEY

# Model -> Provider mapping
models:
  # Route newer models to Sweden
  gpt-5.1: azure_sweden
  gpt-5-pro: azure_sweden

  # Route standard models to West Europe
  gpt-5-mini: azure_west_europe
  gpt-4o: azure_west_europe
  gpt-4o-mini: azure_west_europe

  # Embeddings
  text-embedding-3-small: azure_west_europe
  text-embedding-3-large: azure_west_europe

# Default for unknown models
default: azure_west_europe
```

### How Routing Works

When the code requests a model (e.g., `gpt-5.1`):

1. Looks up model in `llm_endpoints.yaml` → finds `azure_sweden`
2. Gets provider config → endpoint from `AZURE_OPENAI_SWEDEN_ENDPOINT`
3. Makes API call to that endpoint

This is transparent to the rest of the codebase.

---

## Embedding Configuration

Embeddings are used for semantic email search. Configure which provider to use:

```bash
# .env
EMBEDDING_PROVIDER=azure  # or "openai"
```

The embedding model is determined by the provider:
- **OpenAI**: Uses `text-embedding-3-small` by default
- **Azure**: Uses the model deployed at your endpoint

### Azure Embedding Setup

1. Deploy an embedding model in Azure OpenAI Studio
2. Note the deployment name (e.g., `text-embedding-3-small`)
3. Add to `llm_endpoints.yaml`:

```yaml
models:
  text-embedding-3-small: azure_west_europe
```

---

## Model Selection

Different tasks use different models. Configure in your processing scripts or use defaults:

| Task | Default Model | Environment Override |
|------|---------------|---------------------|
| Email classification | `gpt-4o-mini` | Set in classifier config |
| Two-stage classification | `gpt-4o-mini` → `gpt-4o` | Configurable |
| Embeddings | `text-embedding-3-small` | `EMBEDDING_PROVIDER` |
| Name extraction | `gpt-4o-mini` | `AZURE_OPENAI_NAME_EXTRACT_MODEL` |

---

## Full Example: Mixed Provider Setup

This example uses Azure for most tasks, OpenAI as fallback, and Anthropic for specific use cases:

**`.env`:**
```bash
# Azure (primary)
AZURE_OPENAI_API_KEY=azure-key-here
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# Azure Sweden (for newer models)
AZURE_OPENAI_SWEDEN_API_KEY=sweden-key-here
AZURE_OPENAI_SWEDEN_ENDPOINT=https://my-sweden-resource.openai.azure.com

# OpenAI (fallback)
OPENAI_API_KEY=sk-proj-...

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Use Azure for embeddings
EMBEDDING_PROVIDER=azure
```

**`llm_endpoints.yaml`:**
```yaml
providers:
  azure_west_europe:
    type: azure
    endpoint_env: AZURE_OPENAI_ENDPOINT
    api_key_env: AZURE_OPENAI_API_KEY
    api_version: "2025-04-01-preview"

  azure_sweden:
    type: azure
    endpoint_env: AZURE_OPENAI_SWEDEN_ENDPOINT
    api_key_env: AZURE_OPENAI_SWEDEN_API_KEY
    api_version: "2025-04-01-preview"

  openai:
    type: openai
    api_key_env: OPENAI_API_KEY

  anthropic:
    type: anthropic
    api_key_env: ANTHROPIC_API_KEY

models:
  # Premium models → Sweden
  gpt-5.1: azure_sweden
  gpt-5-pro: azure_sweden

  # Standard models → West Europe
  gpt-4o: azure_west_europe
  gpt-4o-mini: azure_west_europe
  gpt-5-mini: azure_west_europe

  # Embeddings → West Europe
  text-embedding-3-small: azure_west_europe
  text-embedding-3-large: azure_west_europe

  # Claude models → Anthropic
  claude-3-5-sonnet-20241022: anthropic
  claude-3-haiku-20240307: anthropic

default: azure_west_europe
```

---

## Ollama (Local LLM)

For fully local processing without external API calls:

**`.env`:**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

**`llm_endpoints.yaml`:**
```yaml
providers:
  ollama:
    type: ollama
    base_url: "http://localhost:11434"

models:
  llama3: ollama
  mistral: ollama
  codellama: ollama

default: ollama
```

**Start Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Verify it's running
curl http://localhost:11434/api/tags
```

---

## Troubleshooting

### "Model not found" errors

1. Check the model is deployed in Azure OpenAI Studio
2. Verify the deployment name matches what's in `llm_endpoints.yaml`
3. Ensure the endpoint URL is correct

### "Authentication failed"

1. Verify API keys are set in `.env`
2. Check the environment variable names match `llm_endpoints.yaml`
3. Ensure keys haven't expired or been rotated

### Slow responses

1. Check if you're routing to a distant region
2. Consider using a faster model (e.g., `gpt-4o-mini` instead of `gpt-4o`)
3. For embeddings, `text-embedding-3-small` is faster than `large`

### Testing your configuration

```bash
# Test classification
python3 process_inbox.py --account work --dry-run --limit 1

# Check which provider is being used (enable debug logging)
LOG_LEVEL=DEBUG python3 process_inbox.py --account work --dry-run --limit 1
```

---

## Cost Optimization

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| Use `gpt-4o-mini` for classification | ~90% vs gpt-4o | Slightly less accurate |
| Use two-stage classification | ~70% | Mini handles easy cases, full model for complex |
| Use `text-embedding-3-small` | ~80% vs large | Minimal quality difference for email search |
| Batch processing | Varies | Latency increase |

### Two-Stage Classification

Process simple emails with a fast model, escalate complex ones:

```bash
python3 process_inbox.py --use-two-stage --dry-run
```

This uses `gpt-4o-mini` for initial classification, only calling `gpt-4o` for uncertain cases.
