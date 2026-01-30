"""
YAML-based prompt loader with include support and variable substitution.

Features:
- !include directive for modular prompt files
- Variable substitution with {variable_name} syntax
- CONFIG_DIR overlay support (personal config overrides defaults)
- Caching for performance

Usage:
    from backend.core.prompt_loader import get_prompt, get_prompt_variables

    # Get a formatted prompt
    prompt = get_prompt("classifier.system_prompt")

    # Get with additional runtime variables
    prompt = get_prompt("classifier.system_prompt", email_subject="Hello")

    # Get all variables
    variables = get_prompt_variables()
"""
import os
import re
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

from backend.core.paths import get_config_path, CONFIG_DIR

logger = logging.getLogger(__name__)

# Cache for loaded prompts
_prompts_cache: Optional[Dict[str, Any]] = None
_prompts_cache_path: Optional[Path] = None


class IncludeLoader(yaml.SafeLoader):
    """YAML loader with !include support for modular config files."""
    pass


def _include_constructor(loader: IncludeLoader, node: yaml.Node) -> Any:
    """Handle !include directive to load external YAML files."""
    # Get the path relative to the current file
    include_path = loader.construct_scalar(node)

    # Resolve relative to the directory containing the current file
    if hasattr(loader, '_root_dir'):
        base_dir = loader._root_dir
    else:
        base_dir = Path.cwd()

    # First try CONFIG_DIR (for overlay), then base_dir
    full_path = None

    # Check CONFIG_DIR first (overlay)
    config_dir = Path(os.getenv("CONFIG_DIR", ""))
    if config_dir.exists():
        overlay_path = config_dir / include_path
        if overlay_path.exists():
            full_path = overlay_path
            logger.debug(f"!include resolved to overlay: {full_path}")

    # Fall back to same directory as parent file
    if full_path is None:
        full_path = base_dir / include_path
        if not full_path.exists():
            # Try prompts subdirectory
            full_path = base_dir / "prompts" / include_path

    if not full_path.exists():
        logger.warning(f"!include file not found: {include_path} (searched {base_dir})")
        return {}

    logger.debug(f"!include loading: {full_path}")

    with open(full_path, 'r') as f:
        # Create a new loader for the included file
        included_loader = IncludeLoader(f)
        included_loader._root_dir = full_path.parent
        try:
            return included_loader.get_single_data()
        finally:
            included_loader.dispose()


# Register the !include constructor
IncludeLoader.add_constructor('!include', _include_constructor)


def _load_prompts_yaml(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load the main prompts.yaml file with caching.

    Resolution order:
    1. CONFIG_DIR/prompts.yaml (overlay)
    2. Default config/prompts.yaml
    """
    global _prompts_cache, _prompts_cache_path

    prompts_path = get_config_path("prompts.yaml")

    if prompts_path is None:
        logger.warning("prompts.yaml not found, using empty config")
        return {"variables": {}, "prompts": {}}

    # Return cached if same path and not forcing reload
    if not force_reload and _prompts_cache is not None and _prompts_cache_path == prompts_path:
        return _prompts_cache

    logger.info(f"Loading prompts from: {prompts_path}")

    with open(prompts_path, 'r') as f:
        loader = IncludeLoader(f)
        loader._root_dir = prompts_path.parent
        try:
            data = loader.get_single_data() or {}
        finally:
            loader.dispose()

    _prompts_cache = data
    _prompts_cache_path = prompts_path

    return data


def get_prompt_variables() -> Dict[str, str]:
    """
    Get the variables section from prompts.yaml.

    Returns:
        Dict of variable_name -> value
    """
    data = _load_prompts_yaml()
    return data.get("variables", {})


def _substitute_variables(template: str, variables: Dict[str, Any]) -> str:
    """
    Substitute {variable_name} placeholders in template.

    Uses a safe substitution that leaves unmatched placeholders as-is.
    """
    def replace(match):
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        # Leave unmatched placeholders for runtime substitution
        return match.group(0)

    # Match {variable_name} but not {{escaped}}
    pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
    return re.sub(pattern, replace, template)


def get_prompt(
    key: str,
    default: Optional[str] = None,
    **runtime_vars
) -> Optional[str]:
    """
    Get a prompt by dotted key path with variable substitution.

    Args:
        key: Dotted path to prompt (e.g., "classifier.system_prompt")
        default: Default value if key not found
        **runtime_vars: Additional variables for substitution

    Returns:
        Formatted prompt string, or default if not found

    Example:
        prompt = get_prompt(
            "classifier.system_prompt",
            email_subject="Hello World",
            sender_name="John Doe"
        )
    """
    data = _load_prompts_yaml()

    # Navigate dotted path
    value = data.get("prompts", {})
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    if value is None:
        if default is not None:
            return default
        logger.warning(f"Prompt key not found: {key}")
        return None

    if not isinstance(value, str):
        logger.warning(f"Prompt key {key} is not a string: {type(value)}")
        return str(value)

    # Merge variables: config variables + runtime variables (runtime takes precedence)
    variables = get_prompt_variables()
    variables.update(runtime_vars)

    # Substitute variables
    result = _substitute_variables(value, variables)

    return result


def get_prompt_section(key: str) -> Dict[str, Any]:
    """
    Get a section of prompts as a dictionary.

    Useful for getting all prompts in a category.

    Args:
        key: Dotted path to section (e.g., "classifier")

    Returns:
        Dict of prompts in that section
    """
    data = _load_prompts_yaml()

    value = data.get("prompts", {})
    for part in key.split("."):
        if isinstance(value, dict):
            value = value.get(part, {})
        else:
            return {}

    return value if isinstance(value, dict) else {}


def reload_prompts() -> None:
    """Force reload of prompts from disk."""
    global _prompts_cache, _prompts_cache_path
    _prompts_cache = None
    _prompts_cache_path = None
    _load_prompts_yaml(force_reload=True)
    logger.info("Prompts reloaded")


def list_available_prompts() -> Dict[str, Any]:
    """
    List all available prompts for debugging.

    Returns:
        The full prompts structure
    """
    data = _load_prompts_yaml()
    return data.get("prompts", {})
