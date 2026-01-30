"""
LLM Config Loader

Simple helper to get provider credentials for a model.
Reads from llm_endpoints.yaml, falls back to env vars.

Features:
- Config validation on load
- Credential testing
- Thread-safe config loading
"""

import os
import yaml
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thread-safe config loading
_config = None
_config_lock = threading.Lock()
_config_path = None  # Track which file was loaded

# Valid provider types
VALID_PROVIDER_TYPES = {"openai", "azure", "anthropic", "ollama"}


@dataclass
class ValidationError:
    """Config validation error."""
    path: str
    message: str


def _validate_config(config: Dict) -> List[ValidationError]:
    """
    Validate config structure and values.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    if not isinstance(config, dict):
        return [ValidationError("root", "Config must be a dictionary")]
    
    # Validate providers section
    providers = config.get("providers", {})
    if not isinstance(providers, dict):
        errors.append(ValidationError("providers", "Must be a dictionary"))
    else:
        for name, cfg in providers.items():
            if not isinstance(cfg, dict):
                errors.append(ValidationError(f"providers.{name}", "Must be a dictionary"))
                continue
            
            # Check type
            ptype = cfg.get("type", name)
            if ptype not in VALID_PROVIDER_TYPES:
                errors.append(ValidationError(
                    f"providers.{name}.type",
                    f"Invalid type '{ptype}'. Must be one of: {VALID_PROVIDER_TYPES}"
                ))
            
            # Azure requires endpoint
            if ptype == "azure" and not cfg.get("endpoint"):
                errors.append(ValidationError(
                    f"providers.{name}.endpoint",
                    "Azure provider requires 'endpoint'"
                ))
            
            # Check api_key_env is a string if present
            if "api_key_env" in cfg and not isinstance(cfg["api_key_env"], str):
                errors.append(ValidationError(
                    f"providers.{name}.api_key_env",
                    "Must be a string (env var name)"
                ))
    
    # Validate models section
    models = config.get("models", {})
    if not isinstance(models, dict):
        errors.append(ValidationError("models", "Must be a dictionary"))
    else:
        for model_name, provider_name in models.items():
            if not isinstance(provider_name, str):
                errors.append(ValidationError(
                    f"models.{model_name}",
                    f"Must be a string (provider name), got {type(provider_name).__name__}"
                ))
            elif provider_name not in providers and provider_name not in VALID_PROVIDER_TYPES:
                errors.append(ValidationError(
                    f"models.{model_name}",
                    f"References unknown provider '{provider_name}'"
                ))
    
    # Validate default
    default = config.get("default")
    if default is not None:
        if not isinstance(default, str):
            errors.append(ValidationError("default", "Must be a string"))
        elif default not in providers and default not in VALID_PROVIDER_TYPES:
            errors.append(ValidationError(
                "default",
                f"References unknown provider '{default}'"
            ))
    
    return errors


def _load_config(force_reload: bool = False) -> Dict:
    """Load config once (thread-safe)."""
    global _config, _config_path
    
    with _config_lock:
        if _config is not None and not force_reload:
            return _config
        
        paths = [
            Path(__file__).parent / "config" / "llm_endpoints.local.yaml",
            Path(__file__).parent / "config" / "llm_endpoints.yaml",
        ]
        
        for path in paths:
            if path.exists():
                try:
                    with open(path) as f:
                        config = yaml.safe_load(f) or {}
                    
                    # Validate
                    errors = _validate_config(config)
                    if errors:
                        error_msgs = [f"  - {e.path}: {e.message}" for e in errors]
                        logger.error(f"Config validation failed ({path}):\n" + "\n".join(error_msgs))
                        raise ValueError(f"Invalid config: {len(errors)} error(s)")
                    
                    _config = config
                    _config_path = path
                    logger.info(f"Loaded and validated LLM config from {path}")
                    return _config
                    
                except yaml.YAMLError as e:
                    logger.error(f"YAML parse error in {path}: {e}")
                    raise
                except ValueError:
                    raise  # Re-raise validation errors
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        
        # No config file - use empty (will fall back to env vars)
        logger.info("No llm_endpoints.yaml found, using environment variables only")
        _config = {}
        _config_path = None
        return _config


def reload_config() -> Dict:
    """Force reload config (e.g., after file changes)."""
    return _load_config(force_reload=True)


def get_config_path() -> Optional[Path]:
    """Get path of loaded config file."""
    _load_config()  # Ensure loaded
    return _config_path


def get_model_config(model_name: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Get provider config for a model.
    
    Args:
        model_name: Model name (e.g., "gpt-5.1", "claude-3-haiku-20240307")
    
    Returns:
        (provider_type, api_key, endpoint, api_version)
        
    Example:
        provider, api_key, endpoint, api_version = get_model_config("gpt-5.1")
        if provider == "azure":
            client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    """
    config = _load_config()
    
    # Look up provider name for this model
    provider_name = config.get("models", {}).get(model_name, config.get("default", "openai"))
    
    # Get provider config
    provider_cfg = config.get("providers", {}).get(provider_name, {})
    provider_type = provider_cfg.get("type", provider_name)
    
    # Resolve credentials based on provider type
    if provider_type == "azure":
        api_key_env = provider_cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        # Support endpoint_env for environment variable reference, or direct endpoint, or fallback
        endpoint_env = provider_cfg.get("endpoint_env")
        if endpoint_env:
            endpoint = os.getenv(endpoint_env)
        else:
            endpoint = provider_cfg.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = provider_cfg.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        return "azure", api_key, endpoint, api_version
    
    elif provider_type == "anthropic":
        api_key_env = provider_cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.getenv(api_key_env)
        return "anthropic", api_key, None, None
    
    elif provider_type == "ollama":
        base_url = provider_cfg.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return "ollama", None, base_url, None
    
    else:  # openai
        api_key_env = provider_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        base_url = provider_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL")
        return "openai", api_key, base_url, None


def list_configured_models() -> Dict[str, str]:
    """List all model -> provider mappings from config."""
    config = _load_config()
    return config.get("models", {})


def list_providers() -> Dict[str, Dict]:
    """List all provider profiles from config."""
    config = _load_config()
    return config.get("providers", {})


# =============================================================================
# Credential Testing
# =============================================================================

@dataclass
class TestResult:
    """Result of a credential/connectivity test."""
    model: str
    provider: str
    success: bool
    message: str
    response: Optional[str] = None  # First few chars of response
    latency_ms: Optional[float] = None


def test_model_credentials(model_name: str, send_test_message: bool = False) -> TestResult:
    """
    Test credentials for a specific model.
    
    Args:
        model_name: Model to test
        send_test_message: If True, sends a small API request to verify connectivity
    
    Returns:
        TestResult with success/failure info
    """
    import time
    
    try:
        provider, api_key, endpoint, api_version = get_model_config(model_name)
    except Exception as e:
        return TestResult(
            model=model_name,
            provider="unknown",
            success=False,
            message=f"Config error: {e}"
        )
    
    # Check credentials are present
    if provider in ("openai", "azure", "anthropic") and not api_key:
        return TestResult(
            model=model_name,
            provider=provider,
            success=False,
            message=f"API key not set (check environment variable)"
        )
    
    if provider == "azure" and not endpoint:
        return TestResult(
            model=model_name,
            provider=provider,
            success=False,
            message="Azure endpoint not configured"
        )
    
    # Credentials look good
    if not send_test_message:
        return TestResult(
            model=model_name,
            provider=provider,
            success=True,
            message=f"Credentials configured (endpoint: {endpoint[:50] + '...' if endpoint and len(endpoint) > 50 else endpoint})"
        )
    
    # Send test message
    try:
        start = time.time()
        
        if provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
                max_tokens=10
            )
            reply = response.choices[0].message.content
            
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=endpoint)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
                max_tokens=10
            )
            reply = response.choices[0].message.content
            
        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'hello' and nothing else."}]
            )
            reply = response.content[0].text
            
        elif provider == "ollama":
            import httpx
            resp = httpx.post(
                f"{endpoint}/api/generate",
                json={"model": model_name, "prompt": "Say hello", "stream": False},
                timeout=30
            )
            resp.raise_for_status()
            reply = resp.json().get("response", "")[:50]
        else:
            return TestResult(
                model=model_name,
                provider=provider,
                success=False,
                message=f"Unknown provider type: {provider}"
            )
        
        latency = (time.time() - start) * 1000
        
        return TestResult(
            model=model_name,
            provider=provider,
            success=True,
            message="API call successful",
            response=reply[:50] if reply else None,
            latency_ms=round(latency, 1)
        )
        
    except Exception as e:
        return TestResult(
            model=model_name,
            provider=provider,
            success=False,
            message=f"API error: {type(e).__name__}: {str(e)[:100]}"
        )


def test_embedding_model(model_name: str) -> TestResult:
    """Test an embedding model with a small request."""
    import time
    
    try:
        provider, api_key, endpoint, api_version = get_model_config(model_name)
    except Exception as e:
        return TestResult(
            model=model_name,
            provider="unknown",
            success=False,
            message=f"Config error: {e}"
        )
    
    if provider in ("openai", "azure") and not api_key:
        return TestResult(
            model=model_name,
            provider=provider,
            success=False,
            message="API key not set"
        )
    
    try:
        start = time.time()
        
        if provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=endpoint)
        
        response = client.embeddings.create(
            model=model_name,
            input="test"
        )
        
        latency = (time.time() - start) * 1000
        dims = len(response.data[0].embedding)
        
        return TestResult(
            model=model_name,
            provider=provider,
            success=True,
            message=f"Embedding successful ({dims} dimensions)",
            latency_ms=round(latency, 1)
        )
        
    except Exception as e:
        return TestResult(
            model=model_name,
            provider=provider,
            success=False,
            message=f"API error: {type(e).__name__}: {str(e)[:100]}"
        )


def test_all_configured_models(send_test_message: bool = False) -> List[TestResult]:
    """
    Test all models in config.
    
    Args:
        send_test_message: If True, sends actual API requests (costs tokens!)
    
    Returns:
        List of TestResults
    """
    results = []
    models = list_configured_models()
    
    for model_name in models:
        if "embedding" in model_name.lower():
            if send_test_message:
                results.append(test_embedding_model(model_name))
            else:
                results.append(test_model_credentials(model_name, send_test_message=False))
        else:
            results.append(test_model_credentials(model_name, send_test_message=send_test_message))
    
    return results


def print_test_results(results: List[TestResult]):
    """Pretty print test results."""
    print("\n" + "=" * 70)
    print("LLM Configuration Test Results")
    print("=" * 70)
    
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"\n{status} {r.model}")
        print(f"  Provider: {r.provider}")
        print(f"  Status: {r.message}")
        if r.response:
            print(f"  Response: {r.response}")
        if r.latency_ms:
            print(f"  Latency: {r.latency_ms}ms")
    
    # Summary
    success = sum(1 for r in results if r.success)
    total = len(results)
    print("\n" + "-" * 70)
    print(f"Summary: {success}/{total} models configured correctly")
    if success < total:
        failed = [r.model for r in results if not r.success]
        print(f"Failed: {', '.join(failed)}")
    print("=" * 70 + "\n")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM configuration")
    parser.add_argument("--model", "-m", help="Test specific model")
    parser.add_argument("--live", "-l", action="store_true", help="Send live API requests")
    parser.add_argument("--list", action="store_true", help="List configured models")
    args = parser.parse_args()
    
    if args.list:
        print("\nConfigured Models:")
        for model, provider in list_configured_models().items():
            print(f"  {model} -> {provider}")
        print(f"\nConfig file: {get_config_path()}")
        sys.exit(0)
    
    if args.model:
        if "embedding" in args.model.lower():
            result = test_embedding_model(args.model) if args.live else test_model_credentials(args.model)
        else:
            result = test_model_credentials(args.model, send_test_message=args.live)
        print_test_results([result])
    else:
        results = test_all_configured_models(send_test_message=args.live)
        print_test_results(results)
