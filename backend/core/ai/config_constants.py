"""
Configuration constants for LLM Scoring system.

Centralized constants to avoid hardcoded paths and magic values.
"""

from pathlib import Path

# Paths
AI_DIR = Path(__file__).parent
CONFIG_DIR = AI_DIR / "config"
DEFAULT_CONFIG_FILE = "model_routing.yaml"
LOCAL_CONFIG_FILE = "model_routing.local.yaml"

# Config defaults
DEFAULT_PROVIDER = "azure"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TEMPERATURE = 1.0

# Application processing model (single source of truth for reprocess_applications.py)
# Change this to switch models for application reprocessing
APPLICATION_MODEL = "gpt-5.1"  # Production model

# Two-stage models (only used by process_inbox.py with --use-two-stage flag)
TWO_STAGE_FAST_MODEL = "gpt-5-mini"
TWO_STAGE_DETAILED_MODEL = "gpt-5.1"

# Token estimation (improved from crude /4)
# Based on OpenAI's tokenizer rules: ~1 token per 4 chars for English
# More conservative for other languages
CHARS_PER_TOKEN_ENGLISH = 4
CHARS_PER_TOKEN_CONSERVATIVE = 3

# Supported providers
SUPPORTED_PROVIDERS = {"openai", "anthropic", "azure"}

# Temperature bounds
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

# Two-stage classifier thresholds
STAGE_2_CONFIDENCE_THRESHOLD = 0.85  # Below this, use detailed scoring
AUTO_OPT_CONFIDENCE_THRESHOLD = 0.80  # Below this, trigger fallback
PERSISTENT_LOW_CONFIDENCE_THRESHOLD = 0.80  # Both models below this = flag

# A/B testing
AB_TEST_DEFAULT_MODELS = ["gpt-5-mini", "gpt-5.1"]
AB_TEST_TIMEOUT_SECONDS = 30  # Max time per model in A/B test

# File paths
AB_TEST_RESULTS_FILE = "ab_test_results.jsonl"

