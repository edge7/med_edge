from typing import Any, Dict, List, Optional, Union

from loguru import logger

# Define constants first to avoid circular import
SST = "SPEECH_TO_TEXT"
TTS = "TEXT_TO_SPEECH"
GOOGLE_SEARCH = "GOOGLE_SEARCH"
SEARCH_INFO_PER_LEARNING_PLAN = "SEARCH_INFO_PER_LEARNING_PLAN"
IMAGE_ANALYZER = "IMAGE_ANALYZER"
IMAGE_GENERATION = "IMAGE_GENERATION"
GOOGLE_IMAGE_GENERATION = "GOOGLE_IMAGE_GENERATION"
TEXT_PROCESSING = "TEXT_PROCESSING"
CORE_CONVERSATION = "CORE_CONVERSATION"
OPENAI_TTS_PODCAST = "OPENAI_TTS_PODCAST"
EXERCISE_PREPARATION = "EXERCISE_PREPARATION"
COMPACT_CONVERSATION = "COMPACT_CONVERSATION"
MAYBE_GOOGLE_SEARCH_PROFESSOR_HELPER = "MAYBE_GOOGLE_SEARCH_PROF_HELPER"
ACTUAL_GOOGLE_SEARCH_PROFESSOR_HELPER = "ACTUAL_GOOGLE_SEARCH_PROFESSOR_HELPER"
COLLECT_FEEDBACK = "COLLECT_FEEDBACK"
FINAL_WRAP_UP = "FINAL_WRAP_UP"
STATUS_MOOD_SUMMARY = "STATUS_MOOD_SUMMARY"
EXTRACT_SUBTOPIC_AND_BACKGROUND = "EXTRACT_SUBTOPIC_AND_BACKGROUND"
WHICH_STEP_WE_ARE_AT = "WHICH_STEP_WE_ARE_AT"
EMPATHY = "EMPATHY"
SECONDS = "seconds"
MODEL = "model"
NUMBER_OF_CHARS = "number_of_chars"
COST_PER_MINUTE_SST = {"gpt-4o-transcribe": 0.008}
TEXT_COMPLETION = "TEXT_COMPLETION"
COMPLETION_TOKENS = "COMPLETION_TOKENS"
REASONING_TOKENS = "REASONING_TOKENS"
PROMPT_TEXT_TOKENS = "PROMPT_TEXT_TOKEN"
CACHED_PROMPT_TEXT_TOKENS = "CACHED_PROMPT_TEXT_TOKENS"
DETAILS = "DETAILS"
COST_TYPE = "COST_TYPE"
COST_INFO = "COST_INFO"
NUMBER_OF_MINUTES = "NUMBER_OF_MINUTES"

def extract_text_cost_gemini(response, model, details=""):
    """Extract token usage from Gemini response formats."""

    if response is None:
        return None

    # Initialize with zeros
    usage_data = {
        COMPLETION_TOKENS: 0,
        REASONING_TOKENS: 0,
        PROMPT_TEXT_TOKENS: 0,
        CACHED_PROMPT_TEXT_TOKENS: 0,
        DETAILS: "",
        MODEL: "",
    }

    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return usage_data

    usage = response.usage_metadata

    # Gemini specific field mappings
    usage_data[COMPLETION_TOKENS] = getattr(usage, "candidates_token_count", 0)
    usage_data[PROMPT_TEXT_TOKENS] = getattr(usage, "prompt_token_count", 0)
    usage_data[REASONING_TOKENS] = getattr(usage, "thoughts_token_count", 0) or 0

    if usage_data[PROMPT_TEXT_TOKENS] is None:
        logger.error(f"Prompt text tokens is null {details} {model}. Setting it to 0")
        usage_data[PROMPT_TEXT_TOKENS] = 0

    # Check for cached tokens (might be in cache_tokens_details or cached_content_token_count)
    usage_data[CACHED_PROMPT_TEXT_TOKENS] = (
        getattr(usage, "cached_content_token_count", 0) or 0
    )

    usage_data[DETAILS] = details
    usage_data[MODEL] = model

    return usage_data


def extract_text_cost_claude(response, model, details=""):
    """Extract token usage from Claude response formats."""

    if response is None:
        return None

    # Initialize with zeros
    usage_data = {
        COMPLETION_TOKENS: 0,
        REASONING_TOKENS: 0,
        PROMPT_TEXT_TOKENS: 0,
        CACHED_PROMPT_TEXT_TOKENS: 0,
        DETAILS: "",
        MODEL: "",
    }

    if not hasattr(response, "usage"):
        return usage_data

    usage = response.usage

    # Claude specific field mappings
    usage_data[COMPLETION_TOKENS] = getattr(usage, "output_tokens", 0)
    usage_data[PROMPT_TEXT_TOKENS] = getattr(usage, "input_tokens", 0)
    usage_data[REASONING_TOKENS] = (
        getattr(usage, "thoughts_token_count", 0) or 0
    )  # Am not using thinking for claude

    usage_data[CACHED_PROMPT_TEXT_TOKENS] = (
        getattr(usage, "cache_read_input_tokens", 0) or 0
    )

    usage_data[DETAILS] = details
    usage_data[MODEL] = model

    return usage_data


def extract_text_cost_open_ai_grok(response, model, details=""):
    """Extract token usage from various provider response formats."""

    if response is None:
        return None
    DETAILED_INFO = details
    # Initialize with zeros
    usage_data = {
        COMPLETION_TOKENS: 0,
        REASONING_TOKENS: 0,
        PROMPT_TEXT_TOKENS: 0,
        CACHED_PROMPT_TEXT_TOKENS: 0,
        DETAILS: 0,
        MODEL: "",
    }

    if not hasattr(response, "usage") or not response.usage:
        logger.error("Response usage is not available!")
        return usage_data

    usage = response.usage

    # Basic token counts - prioritize specific names
    usage_data[COMPLETION_TOKENS] = getattr(usage, "completion_tokens", 0)
    usage_data[PROMPT_TEXT_TOKENS] = getattr(usage, "prompt_text_tokens", 0) or getattr(
        usage, "prompt_tokens", 0
    )

    # Try direct attributes first
    usage_data[REASONING_TOKENS] = getattr(usage, "reasoning_tokens", 0) or 0
    usage_data[CACHED_PROMPT_TEXT_TOKENS] = getattr(
        usage, "cached_prompt_text_tokens", 0
    )
    if usage_data[CACHED_PROMPT_TEXT_TOKENS] == 0:
        usage_data[CACHED_PROMPT_TEXT_TOKENS] = getattr(
            usage, "cached_prompt_tokens", 0
        )

    # Try nested completion_tokens_details
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        details = usage.completion_tokens_details
        usage_data[REASONING_TOKENS] = (
            max(
                usage_data[REASONING_TOKENS],
                getattr(details, "reasoning_tokens", 0),
            )
            or 0
        )

    # Try nested prompt_tokens_details
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        usage_data[CACHED_PROMPT_TEXT_TOKENS] = max(
            usage_data[CACHED_PROMPT_TEXT_TOKENS],
            getattr(details, "cached_tokens", 0),
        )

    # Handle other provider formats (Anthropic, Azure, etc.)
    if not usage_data[COMPLETION_TOKENS]:
        usage_data[COMPLETION_TOKENS] = getattr(usage, "output_tokens", 0)

    if not usage_data[PROMPT_TEXT_TOKENS]:
        usage_data[PROMPT_TEXT_TOKENS] = getattr(usage, "input_tokens", 0)

    # Some providers might have different cache field names
    if not usage_data[CACHED_PROMPT_TEXT_TOKENS]:
        usage_data[CACHED_PROMPT_TEXT_TOKENS] = getattr(
            usage, "cache_read_input_tokens", 0
        )
    usage_data[DETAILS] = DETAILED_INFO
    usage_data[MODEL] = model
    return usage_data


def extract_cost_text(response, model, details=""):
    if response is None:
        return None
    if "google" in model.lower():
        return extract_text_cost_gemini(response, model, details)
    if "claude" in model.lower():
        return extract_text_cost_claude(response, model, details)
    return extract_text_cost_open_ai_grok(response, model, details)

def build_response(
    real_result: Any,
    unique_id: str,
    cost: Optional[Union[Dict, List[Dict]]] = None,
) -> Dict[str, Any]:
    """
    Creates a standardized dictionary for functions using the @log_cost decorator.

    This utility ensures the return value is always in the format expected
    by the decorator, promoting consistency and making future changes easy.

    Args:
        real_result: The actual result the decorated function should ultimately return.
        unique_id: The identifier associated with the operation for logging.
        cost: The calculated cost(s). Can be a single dict or a list of dict

    Returns:
        A dictionary structured for the @log_cost decorator to process.
    """
    if cost is None or unique_id is None:
        return real_result
    return {
        "real_result": real_result,
        "cost": {"unique_id": unique_id, "value": cost},
    }

# Import cost writer functions after constants are defined to avoid circular import
from .cost_writer import (
    initialize_cost_writer,
    shutdown_cost_writer,
    push_cost,
    get_csv_path,
    is_initialized,
)