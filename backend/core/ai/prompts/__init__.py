"""
AI Prompt Management with Versioning

Centralized prompt storage with version control for A/B testing and tracking.

v3.0: Added category-specific prompt registry with auto-discovery.
"""
from .classifier_prompts import (
    get_classifier_prompt,
    CURRENT_VERSION,
    AVAILABLE_VERSIONS
)

# v3.0: Prompt Registry
from typing import Callable, Dict, Optional
from fnmatch import fnmatch
import logging
import pkgutil
import importlib

logger = logging.getLogger(__name__)

# Global registry: category_pattern → prompt_function
prompt_registry: Dict[str, Callable] = {}


def register_prompt(category_pattern: str):
    """
    Decorator to register a prompt function for a category pattern.
    
    Args:
        category_pattern: Category pattern (e.g., "application-*", "invitation-speaking")
    
    Example:
        @register_prompt("application-*")
        def build_application_prompt(email, sender_history=None):
            return f"Evaluate this application from {email.from_address}..."
    """
    def decorator(func: Callable):
        prompt_registry[category_pattern] = func
        logger.info(f"Registered prompt for pattern: {category_pattern}")
        return func
    return decorator


def get_prompt_for_category(category: str) -> Optional[Callable]:
    """
    Get the prompt function for a category.
    
    Matching order:
    1. Exact match
    2. Wildcard pattern match (most specific first)
    3. None (use default)
    
    Args:
        category: Email category (e.g., "application-phd")
        
    Returns:
        Prompt function or None if no match
    """
    # Check exact match first
    if category in prompt_registry:
        logger.debug(f"Exact prompt match for {category}")
        return prompt_registry[category]
    
    # Check wildcard patterns (sort by specificity)
    patterns = [(k, v) for k, v in prompt_registry.items() if "*" in k]
    patterns.sort(key=lambda x: x[0].count("*"))  # Fewer wildcards = more specific
    
    for pattern, func in patterns:
        if fnmatch(category, pattern):
            logger.debug(f"Prompt pattern match: {category} → {pattern}")
            return func
    
    logger.debug(f"No prompt match for {category}, using default")
    return None


# Auto-discover and import all prompt modules in this directory
# This ensures all @register_prompt decorators are executed on startup
def _auto_discover_prompts():
    """Auto-import all modules in the prompts package."""
    package_dir = __path__[0]
    logger.info(f"Auto-discovering prompts in {package_dir}")
    
    # Exclude non-Python modules and example/documentation files
    excluded_modules = ["__init__", "classifier_prompts", "application_examples"]
    
    for _, name, is_pkg in pkgutil.iter_modules(__path__):
        if not is_pkg and name not in excluded_modules:
            try:
                module = importlib.import_module(f".{name}", __name__)
                logger.info(f"Loaded prompt module: {name}")
            except Exception as e:
                logger.error(f"Failed to load prompt module {name}: {e}")


# Run auto-discovery when this module is imported
_auto_discover_prompts()


__all__ = [
    'get_classifier_prompt',
    'CURRENT_VERSION',
    'AVAILABLE_VERSIONS',
    'register_prompt',
    'get_prompt_for_category',
    'prompt_registry',
]

