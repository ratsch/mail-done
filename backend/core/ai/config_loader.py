"""
Configuration Loader for Modular LLM Scoring

Handles loading, validation, and category matching for model routing configuration.
"""

import yaml
import os
from pathlib import Path
from fnmatch import fnmatch
from typing import Dict, Optional
import logging

from .config_constants import (
    CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    LOCAL_CONFIG_FILE,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    SUPPORTED_PROVIDERS,
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates model routing configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to model_routing.yaml (optional)
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load and validate model routing configuration from YAML."""
        if config_path is None:
            # Use constants instead of hardcoded paths
            config_path = CONFIG_DIR / DEFAULT_CONFIG_FILE
        else:
            config_path = Path(config_path)
        
        # Check for local override
        if config_path.name == DEFAULT_CONFIG_FILE:
            local_path = CONFIG_DIR / LOCAL_CONFIG_FILE
            if local_path.exists():
                logger.info(f"Using local config override: {local_path}")
                config_path = local_path
        
        if not config_path.exists():
            # WARNING level (was incorrectly using WARNING in message but INFO level)
            logger.warning(
                f"Config file not found at {config_path}. "
                f"Using default configuration (provider={DEFAULT_PROVIDER}, model={DEFAULT_MODEL})"
            )
            return {
                "default": {
                    "provider": DEFAULT_PROVIDER,
                    "model": DEFAULT_MODEL,
                    "temperature": DEFAULT_TEMPERATURE
                }
            }
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config at {config_path}: {e}")
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading config {config_path}: {e}")
            raise
        
        # Handle empty config
        if not config:
            logger.warning(f"Config file {config_path} is empty, using defaults")
            return {
                "default": {
                    "provider": DEFAULT_PROVIDER,
                    "model": DEFAULT_MODEL,
                    "temperature": DEFAULT_TEMPERATURE
                }
            }
        
        # Validate schema
        self._validate_config(config)
        
        logger.debug(f"Successfully loaded config from {config_path}")
        return config
    
    def _validate_config(self, config: dict):
        """Validate configuration schema."""
        try:
            from jsonschema import validate, ValidationError
        except ImportError:
            # CHANGED: Make this an error, not a warning - validation is critical
            logger.error(
                "jsonschema library not installed. Install with: pip install jsonschema"
            )
            raise ImportError(
                "jsonschema is required for config validation. "
                "Install it with: pip install jsonschema"
            )
        
        # Use constants for validation
        # Note: provider is optional - auto-detected from llm_endpoints.yaml
        schema = {
            "type": "object",
            "properties": {
                "default": {
                    "type": "object",
                    "required": ["model"],  # provider is optional - auto-detected from llm_endpoints.yaml
                    "properties": {
                        "provider": {
                            "type": "string"
                            # Don't restrict to SUPPORTED_PROVIDERS - azure is also valid
                        },
                        "model": {"type": "string"},
                        "temperature": {
                            "type": "number",
                            "minimum": MIN_TEMPERATURE,
                            "maximum": MAX_TEMPERATURE
                        }
                    }
                },
                "categories": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["model"],  # provider is optional - auto-detected from llm_endpoints.yaml
                        "properties": {
                            "provider": {
                                "type": "string"
                                # Don't restrict to SUPPORTED_PROVIDERS - azure is also valid
                            },
                            "model": {"type": "string"},
                            "temperature": {
                                "type": "number",
                                "minimum": MIN_TEMPERATURE,
                                "maximum": MAX_TEMPERATURE
                            }
                        }
                    }
                },
                "env_overrides": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "patterns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["category_prefix", "env_var"],
                                "properties": {
                                    "category_prefix": {"type": "string"},
                                    "env_var": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            "required": ["default"]
        }
        
        try:
            validate(instance=config, schema=schema)
            logger.debug("Config validation passed")
        except ValidationError as e:
            logger.error(f"Invalid config schema: {e.message}")
            raise ValueError(
                f"Invalid model_routing.yaml schema: {e.message}\n"
                f"Path: {' -> '.join(str(p) for p in e.path) if e.path else 'root'}"
            )
    
    def get_config_for_category(self, category: str) -> dict:
        """
        Get model configuration for a category.
        
        Matching order:
        1. Environment variable override (if enabled)
        2. Exact category match
        3. Wildcard pattern match (ordered by specificity)
        4. Default configuration
        
        Args:
            category: Email category (e.g., "application-phd")
            
        Returns:
            Dict with provider, model, temperature keys
        """
        match_type = None
        
        # Check environment variable overrides
        if self.config.get("env_overrides", {}).get("enabled", False):
            for pattern in self.config["env_overrides"].get("patterns", []):
                prefix = pattern["category_prefix"]
                if category.startswith(prefix):
                    env_var = pattern["env_var"]
                    env_model = os.getenv(env_var)
                    if env_model:
                        match_type = f"env override ({env_var})"
                        config = {
                            "provider": os.getenv(
                                f"{prefix.upper()}_LLM_PROVIDER",
                                "openai"
                            ),
                            "model": env_model,
                            "temperature": float(os.getenv(
                                f"{prefix.upper()}_LLM_TEMPERATURE",
                                "0.1"
                            ))
                        }
                        logger.info(
                            f"Model for category '{category}' selected via {match_type}: "
                            f"Provider={config['provider']}, Model={config['model']}"
                        )
                        return config
        
        # Check exact match first
        categories = self.config.get("categories", {})
        if category in categories:
            match_type = "exact match"
            config = categories[category]
        else:
            # Check wildcard patterns (sort by specificity)
            patterns = [(k, v) for k, v in categories.items() if "*" in k]
            # Fewer wildcards = more specific
            patterns.sort(key=lambda x: x[0].count("*"))
            
            for pattern, pattern_config in patterns:
                if fnmatch(category, pattern):
                    match_type = f"pattern match ({pattern})"
                    config = pattern_config
                    break
            else:
                # Fallback to default
                match_type = "default"
                config = self.config.get("default", {
                    "model": "gpt-5-mini",
                    "temperature": 1.0
                })
        
        logger.info(
            f"Model for category '{category}' selected via {match_type}: "
            f"Provider={config.get('provider')}, Model={config.get('model')}"
        )
        return config
    
    def get_default_config(self) -> dict:
        """Get the default configuration."""
        return self.config.get("default", {
            "model": "gpt-5-mini",
            "temperature": 1.0
        })

