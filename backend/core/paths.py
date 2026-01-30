"""
Centralized path configuration for mail-done.

Supports:
- Local config: ./config/*.yaml
- External overlay: CONFIG_DIR=/path/to/private/config
- Fallback to .example.yaml when .yaml missing

Usage:
    from backend.core.paths import get_config_path, CONFIG_DIR, PROMPTS_DIR

    # Get config file with fallback
    rules_path = get_config_path("classification_rules.yaml")

    # Required config (raises if not found)
    accounts_path = get_config_path("accounts.yaml", required=True)
"""
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Determine repo root (where process_inbox.py lives)
# From backend/core/paths.py -> backend/core -> backend -> email-processor (or repo root)
_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG_DIR = _REPO_ROOT / "config"
_DEFAULT_PROMPTS_DIR = Path(__file__).parent / "ai" / "prompts"

# Environment-configurable paths
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", str(_DEFAULT_CONFIG_DIR)))
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", str(_DEFAULT_PROMPTS_DIR)))


def get_config_path(filename: str, required: bool = False) -> Optional[Path]:
    """
    Resolve config file path with fallback logic.

    Resolution order:
    1. CONFIG_DIR / filename
    2. CONFIG_DIR / filename.example.yaml (if .yaml)
    3. Default config dir / filename
    4. Default config dir / filename.example.yaml

    Args:
        filename: Config filename (e.g., "classification_rules.yaml")
        required: If True, raise FileNotFoundError when not found

    Returns:
        Path to config file, or None if not found and not required

    Raises:
        FileNotFoundError: If required=True and file not found
    """
    candidates = []

    # External config dir (from CONFIG_DIR env var)
    candidates.append(CONFIG_DIR / filename)
    if filename.endswith('.yaml'):
        example_name = filename.replace('.yaml', '.example.yaml')
        candidates.append(CONFIG_DIR / example_name)

    # Default config dir (if different from CONFIG_DIR)
    if CONFIG_DIR != _DEFAULT_CONFIG_DIR:
        candidates.append(_DEFAULT_CONFIG_DIR / filename)
        if filename.endswith('.yaml'):
            example_name = filename.replace('.yaml', '.example.yaml')
            candidates.append(_DEFAULT_CONFIG_DIR / example_name)

    for path in candidates:
        if path.exists():
            logger.debug(f"Config '{filename}' resolved to: {path}")
            return path

    if required:
        searched = [str(c) for c in candidates]
        raise FileNotFoundError(
            f"Required config file '{filename}' not found.\n"
            f"Searched: {searched}\n"
            f"Hint: Copy {filename.replace('.yaml', '.example.yaml')} to {filename} and customize it."
        )

    logger.debug(f"Config '{filename}' not found (optional)")
    return None


def get_prompt_path(filename: str) -> Optional[Path]:
    """
    Get path to a prompt file, checking PROMPTS_DIR first.

    Args:
        filename: Prompt filename (e.g., "production_prompt.py")

    Returns:
        Path to prompt file, or None if not found
    """
    # Check external prompts directory first
    if PROMPTS_DIR != _DEFAULT_PROMPTS_DIR:
        external = PROMPTS_DIR / filename
        if external.exists():
            logger.debug(f"Prompt '{filename}' resolved to overlay: {external}")
            return external

    # Fall back to builtin
    builtin = _DEFAULT_PROMPTS_DIR / filename
    if builtin.exists():
        logger.debug(f"Prompt '{filename}' resolved to builtin: {builtin}")
        return builtin

    logger.debug(f"Prompt '{filename}' not found")
    return None


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return _REPO_ROOT


def is_using_external_config() -> bool:
    """Check if using external CONFIG_DIR."""
    return CONFIG_DIR != _DEFAULT_CONFIG_DIR


def is_using_external_prompts() -> bool:
    """Check if using external PROMPTS_DIR."""
    return PROMPTS_DIR != _DEFAULT_PROMPTS_DIR


# Log configuration on import (only if DEBUG level)
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"paths.py initialized:")
    logger.debug(f"  REPO_ROOT: {_REPO_ROOT}")
    logger.debug(f"  CONFIG_DIR: {CONFIG_DIR} (external: {is_using_external_config()})")
    logger.debug(f"  PROMPTS_DIR: {PROMPTS_DIR} (external: {is_using_external_prompts()})")
