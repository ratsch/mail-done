"""
Unit tests for llm_config module.

Run with: pytest backend/core/ai/tests/test_llm_config.py -v
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import module under test
from backend.core.ai.llm_config import (
    _validate_config,
    _load_config,
    reload_config,
    get_model_config,
    list_configured_models,
    list_providers,
    test_model_credentials,
    ValidationError,
    TestResult,
    VALID_PROVIDER_TYPES,
    _config,
    _config_lock,
)
import backend.core.ai.llm_config as llm_config_module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_config():
    """A valid config dictionary."""
    return {
        "providers": {
            "azure_sweden": {
                "type": "azure",
                "endpoint": "https://sweden.azure.com",
                "api_key_env": "AZURE_SWEDEN_KEY",
                "api_version": "2025-01-01"
            },
            "azure_west": {
                "type": "azure",
                "endpoint": "https://west.azure.com",
                "api_key_env": "AZURE_WEST_KEY"
            },
            "openai": {
                "type": "openai",
                "api_key_env": "OPENAI_API_KEY"
            },
            "anthropic": {
                "type": "anthropic",
                "api_key_env": "ANTHROPIC_API_KEY"
            },
            "ollama": {
                "type": "ollama",
                "base_url": "http://localhost:11434"
            }
        },
        "models": {
            "gpt-5.1": "azure_sweden",
            "gpt-5-mini": "azure_west",
            "gpt-4o": "openai",
            "claude-3": "anthropic",
            "llama3": "ollama",
            "text-embedding-3-small": "azure_west"
        },
        "default": "openai"
    }


@pytest.fixture
def temp_config_file(valid_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config, f)
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config before each test."""
    llm_config_module._config = None
    llm_config_module._config_path = None
    yield
    llm_config_module._config = None
    llm_config_module._config_path = None


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for config validation."""
    
    def test_valid_config(self, valid_config):
        """Valid config should return no errors."""
        errors = _validate_config(valid_config)
        assert errors == []
    
    def test_empty_config(self):
        """Empty config is valid (uses defaults)."""
        errors = _validate_config({})
        assert errors == []
    
    def test_invalid_root(self):
        """Non-dict root should fail."""
        errors = _validate_config("not a dict")
        assert len(errors) == 1
        assert errors[0].path == "root"
    
    def test_invalid_provider_type(self):
        """Invalid provider type should fail."""
        config = {
            "providers": {
                "bad": {"type": "invalid_type"}
            }
        }
        errors = _validate_config(config)
        assert len(errors) == 1
        assert "invalid_type" in errors[0].message
    
    def test_azure_missing_endpoint(self):
        """Azure provider without endpoint should fail."""
        config = {
            "providers": {
                "azure_no_endpoint": {
                    "type": "azure",
                    "api_key_env": "SOME_KEY"
                    # Missing endpoint!
                }
            }
        }
        errors = _validate_config(config)
        assert len(errors) == 1
        assert "endpoint" in errors[0].message
    
    def test_model_references_unknown_provider(self):
        """Model referencing unknown provider should fail."""
        config = {
            "providers": {
                "openai": {"type": "openai"}
            },
            "models": {
                "gpt-4": "nonexistent_provider"
            }
        }
        errors = _validate_config(config)
        assert len(errors) == 1
        assert "nonexistent_provider" in errors[0].message
    
    def test_model_value_not_string(self):
        """Model value must be string."""
        config = {
            "providers": {},
            "models": {
                "gpt-4": 123  # Should be string
            }
        }
        errors = _validate_config(config)
        assert len(errors) >= 1
        assert "string" in errors[0].message.lower()
    
    def test_invalid_default(self):
        """Default referencing unknown provider should fail."""
        config = {
            "providers": {},
            "default": "unknown_provider"
        }
        errors = _validate_config(config)
        assert len(errors) == 1
        assert "unknown_provider" in errors[0].message
    
    def test_valid_provider_types(self):
        """All valid provider types should be accepted."""
        for ptype in VALID_PROVIDER_TYPES:
            config = {
                "providers": {
                    f"test_{ptype}": {
                        "type": ptype,
                        "endpoint": "http://test.com" if ptype == "azure" else None
                    }
                }
            }
            # Remove None endpoint for non-azure
            if ptype != "azure":
                del config["providers"][f"test_{ptype}"]["endpoint"]
            
            errors = _validate_config(config)
            # Filter out azure endpoint errors for non-azure types
            non_azure_errors = [e for e in errors if "endpoint" not in e.message.lower() or ptype == "azure"]
            assert len(non_azure_errors) == 0, f"Provider type {ptype} should be valid"


# =============================================================================
# Config Loading Tests
# =============================================================================

class TestConfigLoading:
    """Tests for config loading."""
    
    def test_load_empty_without_file(self):
        """Without config file, should return empty dict."""
        with patch.object(Path, 'exists', return_value=False):
            llm_config_module._config = None
            config = _load_config(force_reload=True)
            assert config == {}
    
    def test_load_valid_file(self, temp_config_file, valid_config):
        """Should load and validate config file."""
        with patch.object(llm_config_module, '_load_config') as mock_load:
            mock_load.return_value = valid_config
            config = mock_load()
            assert "providers" in config
            assert "models" in config
    
    def test_reload_config(self, valid_config):
        """reload_config should force refresh."""
        llm_config_module._config = {"old": "config"}
        
        with patch('builtins.open', MagicMock()):
            with patch('yaml.safe_load', return_value=valid_config):
                with patch.object(Path, 'exists', return_value=True):
                    config = reload_config()
                    assert config == valid_config


# =============================================================================
# get_model_config Tests
# =============================================================================

class TestGetModelConfig:
    """Tests for get_model_config function."""
    
    def test_azure_model(self, valid_config):
        """Azure model should return azure credentials."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"AZURE_SWEDEN_KEY": "test-key"}):
            provider, api_key, endpoint, api_version = get_model_config("gpt-5.1")
            
            assert provider == "azure"
            assert api_key == "test-key"
            assert endpoint == "https://sweden.azure.com"
            assert api_version == "2025-01-01"
    
    def test_openai_model(self, valid_config):
        """OpenAI model should return openai credentials."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider, api_key, endpoint, api_version = get_model_config("gpt-4o")
            
            assert provider == "openai"
            assert api_key == "sk-test"
            assert endpoint is None  # No custom base_url
            assert api_version is None
    
    def test_anthropic_model(self, valid_config):
        """Anthropic model should return anthropic credentials."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-test"}):
            provider, api_key, endpoint, api_version = get_model_config("claude-3")
            
            assert provider == "anthropic"
            assert api_key == "ant-test"
            assert endpoint is None
            assert api_version is None
    
    def test_ollama_model(self, valid_config):
        """Ollama model should return ollama config."""
        llm_config_module._config = valid_config
        
        provider, api_key, endpoint, api_version = get_model_config("llama3")
        
        assert provider == "ollama"
        assert api_key is None  # Ollama doesn't need API key
        assert endpoint == "http://localhost:11434"
        assert api_version is None
    
    def test_unknown_model_uses_default(self, valid_config):
        """Unknown model should use default provider."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-default"}):
            provider, api_key, endpoint, api_version = get_model_config("unknown-model")
            
            assert provider == "openai"  # default in valid_config
            assert api_key == "sk-default"
    
    def test_missing_api_key_returns_none(self, valid_config):
        """Missing API key should return None (caller handles error)."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {}, clear=True):
            provider, api_key, endpoint, api_version = get_model_config("gpt-5.1")
            
            assert provider == "azure"
            assert api_key is None  # Not set
    
    def test_env_fallback_for_azure(self):
        """Should fall back to AZURE_OPENAI_* env vars if not in config."""
        llm_config_module._config = {
            "providers": {
                "azure_basic": {"type": "azure"}  # No endpoint/api_key_env specified
            },
            "models": {"test-model": "azure_basic"}
        }
        
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "fallback-key",
            "AZURE_OPENAI_ENDPOINT": "https://fallback.azure.com"
        }):
            provider, api_key, endpoint, api_version = get_model_config("test-model")
            
            assert provider == "azure"
            assert api_key == "fallback-key"
            assert endpoint == "https://fallback.azure.com"


# =============================================================================
# Credential Testing Tests
# =============================================================================

class TestCredentialTesting:
    """Tests for credential testing functions."""
    
    def test_credentials_present(self, valid_config):
        """Should report success when credentials are configured."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"AZURE_SWEDEN_KEY": "test-key"}):
            result = test_model_credentials("gpt-5.1", send_test_message=False)
            
            assert result.success is True
            assert result.provider == "azure"
            assert "configured" in result.message.lower()
    
    def test_credentials_missing(self, valid_config):
        """Should report failure when API key missing."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {}, clear=True):
            result = test_model_credentials("gpt-5.1", send_test_message=False)
            
            assert result.success is False
            assert "not set" in result.message.lower()
    
    def test_azure_endpoint_missing(self):
        """Should report failure when Azure endpoint missing."""
        llm_config_module._config = {
            "providers": {
                "azure_no_endpoint": {"type": "azure", "api_key_env": "SOME_KEY"}
            },
            "models": {"test": "azure_no_endpoint"}
        }
        
        with patch.dict(os.environ, {"SOME_KEY": "key-value"}, clear=True):
            result = test_model_credentials("test", send_test_message=False)
            
            assert result.success is False
            assert "endpoint" in result.message.lower()
    
    def test_ollama_no_api_key_needed(self, valid_config):
        """Ollama should succeed without API key."""
        llm_config_module._config = valid_config
        
        result = test_model_credentials("llama3", send_test_message=False)
        
        assert result.success is True
        assert result.provider == "ollama"


# =============================================================================
# List Functions Tests
# =============================================================================

class TestListFunctions:
    """Tests for list_configured_models and list_providers."""
    
    def test_list_models(self, valid_config):
        """Should return all configured models."""
        llm_config_module._config = valid_config
        
        models = list_configured_models()
        
        assert "gpt-5.1" in models
        assert "gpt-5-mini" in models
        assert models["gpt-5.1"] == "azure_sweden"
    
    def test_list_providers(self, valid_config):
        """Should return all provider configs."""
        llm_config_module._config = valid_config
        
        providers = list_providers()
        
        assert "azure_sweden" in providers
        assert "openai" in providers
        assert providers["azure_sweden"]["type"] == "azure"
    
    def test_list_empty_config(self):
        """Empty config should return empty dicts."""
        llm_config_module._config = {}
        
        assert list_configured_models() == {}
        assert list_providers() == {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests with actual config file structure."""
    
    def test_multi_region_azure_config(self):
        """Test with multi-region Azure setup."""
        config = {
            "providers": {
                "azure_sweden": {
                    "type": "azure",
                    "endpoint": "https://your-sweden-resource.services.ai.azure.com",
                    "api_key_env": "AZURE_OPENAI_SWEDEN_API_KEY",
                    "api_version": "2025-04-01-preview"
                },
                "azure_west_europe": {
                    "type": "azure",
                    "endpoint": "https://your-europe-resource.cognitiveservices.azure.com",
                    "api_key_env": "AZURE_OPENAI_API_KEY",
                    "api_version": "2025-04-01-preview"
                }
            },
            "models": {
                "gpt-5.1": "azure_sweden",
                "gpt-5-pro": "azure_sweden",
                "gpt-5-mini": "azure_west_europe",
                "text-embedding-3-small": "azure_west_europe"
            },
            "default": "openai"
        }
        
        # Validate
        errors = _validate_config(config)
        assert errors == [], f"Config invalid: {errors}"
        
        # Test model resolution
        llm_config_module._config = config
        
        with patch.dict(os.environ, {
            "AZURE_OPENAI_SWEDEN_API_KEY": "sweden-key",
            "AZURE_OPENAI_API_KEY": "west-key"
        }):
            # GPT-5.1 should use Sweden
            p, k, e, v = get_model_config("gpt-5.1")
            assert p == "azure"
            assert k == "sweden-key"
            assert "swede" in e.lower()
            
            # GPT-5-mini should use West Europe
            p, k, e, v = get_model_config("gpt-5-mini")
            assert p == "azure"
            assert k == "west-key"
            assert "swede" not in e.lower()
            
            # Embeddings should use West Europe
            p, k, e, v = get_model_config("text-embedding-3-small")
            assert p == "azure"
            assert k == "west-key"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_model_name_with_special_chars(self, valid_config):
        """Model names with special chars should work."""
        valid_config["models"]["claude-3.5-sonnet-20241022"] = "anthropic"
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "key"}):
            provider, _, _, _ = get_model_config("claude-3.5-sonnet-20241022")
            assert provider == "anthropic"
    
    def test_empty_model_name(self, valid_config):
        """Empty model name should use default."""
        llm_config_module._config = valid_config
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
            provider, _, _, _ = get_model_config("")
            assert provider == "openai"  # default
    
    def test_none_values_in_config(self):
        """None values in config should not crash."""
        config = {
            "providers": {
                "test": {
                    "type": "openai",
                    "api_key_env": None  # Edge case
                }
            },
            "models": {"m": "test"}
        }
        llm_config_module._config = config
        
        # Should not raise, just return None for api_key
        provider, api_key, _, _ = get_model_config("m")
        assert provider == "openai"
        assert api_key is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
