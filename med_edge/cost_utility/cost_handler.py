from loguru import logger
from rich.console import Console
from . import (
    CACHED_PROMPT_TEXT_TOKENS,
    COMPLETION_TOKENS,
    COST_TYPE,
    COST_INFO,
    REASONING_TOKENS,
    PROMPT_TEXT_TOKENS,
    DETAILS,
    MODEL,
    TEXT_PROCESSING,
)


console = Console()
# --- Configuration for persistent storage ---
CSV_FILENAME = "experiment_cost_log.csv"

ONE_MILION = 1000000
text_table_cost = {
    "openai/gpt-4.1": {
        "input": 2 / ONE_MILION,
        "input_cached": 0.5 / ONE_MILION,
        "output": 8 / ONE_MILION,
    },
    "openai/o3": {
        "input": 2 / ONE_MILION,
        "input_cached": 0.5 / ONE_MILION,
        "output": 8 / ONE_MILION,
    },
    "openai/o4-mini": {
        "input": 1.1 / ONE_MILION,
        "input_cached": 0.275 / ONE_MILION,
        "output": 4.4 / ONE_MILION,
    },
    "openai/gpt-5": {
        "input": 1.25 / ONE_MILION,
        "input_cached": 0.125 / ONE_MILION,
        "output": 10 / ONE_MILION,
    },
    "openai/gpt-5-mini": {
        "input": 0.25 / ONE_MILION,
        "input_cached": 0.025 / ONE_MILION,
        "output": 2 / ONE_MILION,
    },
    "openai/gpt-image-1": {
        "input": 10 / ONE_MILION,
        "input_cached": 2.5 / ONE_MILION,
        "output": 40 / ONE_MILION,
    },
    "google/gemini-2.5-pro": {
        "input": 1.5 / ONE_MILION,
        "input_cached": 0.45 / ONE_MILION,
        "output": 12 / ONE_MILION,
    },
    "google/gemini-2.5-flash": {
        "input": 0.3 / ONE_MILION,
        "input_cached": 0.075 / ONE_MILION,
        "output": 2.5 / ONE_MILION,
    },
    "xai/grok-4": {
        "input": 3 / ONE_MILION,
        "input_cached": 0.75 / ONE_MILION,
        "output": 15 / ONE_MILION,
    },
    "xai/grok-3": {
        "input": 3 / ONE_MILION,
        "input_cached": 0.75 / ONE_MILION,
        "output": 15 / ONE_MILION,
    },
    "xai/grok-3-fast": {
        "input": 5 / ONE_MILION,
        "input_cached": 1.25 / ONE_MILION,
        "output": 25 / ONE_MILION,
    },
    "xai/grok-3-mini": {
        "input": 0.3 / ONE_MILION,
        "input_cached": 0.075 / ONE_MILION,
        "output": 0.5 / ONE_MILION,
    },
    "xai/grok-4-fast-non-reasoning": {
        "input": 0.2 / ONE_MILION,
        "input_cached": 0.05 / ONE_MILION,
        "output": 0.5 / ONE_MILION,
    },
    "deepseek/deepseek-chat": {
        "input": 0.56 / ONE_MILION,
        "input_cached": 0.07 / ONE_MILION,
        "output": 1.68 / ONE_MILION,
    },
    "deepseek/deepseek-reasoner": {
        "input": 0.27 * 2 / ONE_MILION,
        "input_cached": 0.07 * 2 / ONE_MILION,
        "output": 1.1 * 2 / ONE_MILION,
    },
    "anthropic/claude-sonnet-4-20250514": {
        "input": 3 / ONE_MILION,
        "input_cached": 0.3 / ONE_MILION,
        "output": 15 / ONE_MILION,
    },
    "anthropic/claude-sonnet-4-5-20250929": {
        "input": 3 / ONE_MILION,
        "input_cached": 0.3 / ONE_MILION,
        "output": 15 / ONE_MILION,
    },
    "sonar-deep-research": {  # Perplexity
        "input": 2 / ONE_MILION,
        "output": 8 / ONE_MILION,
        "citations": 2 / ONE_MILION,
        "search_queries": 5 / 1000,
        "reasoning": 3 / ONE_MILION,
        "request_cost": 0 / 1000,
    },
    "sonar-reasoning-pro": {  # Perplexity
        "input": 2 / ONE_MILION,
        "output": 8 / ONE_MILION,
        "citations": 0 / ONE_MILION,
        "search_queries": 0 / 1000,
        "reasoning": 0 / ONE_MILION,
        "request_cost": 10 / 1000,
    },
}


def tts_cost_table(model):
    if "chirp" in model.lower():  # Google TTS
        return {"cost_per_char": 20 / ONE_MILION}
    raise ValueError(f"Unrecognised tts model: {model}")


def openai_tts(model):
    if model == "gpt-4o-mini-tts":
        return {"cost_per_minute": 0.015}
    raise ValueError(f"Unrecognised tts model: {model}")


def google_image_generation(model):
    if "imagen-4.0-generate-preview-06-06" in model.lower():  # Google Image generation
        return {"cost_per_image": 0.04}
    raise ValueError(f"Unrecognised Image generation model: {model}")


stt_cost_table = {"gpt-4o-transcribe": 0.006}  # Dollar per minute

DB_SCHEMA = {
    "unique_id": None,
    "model": None,
    "cost_input": 0.0,
    "cost_output": 0.0,
    "cost_cached_input": 0.0,
    "details": None,
    "tts_cost": 0.0,
    "sst_cost": 0.0,
    "total_cost": 0.0,
    COST_TYPE: None,
}


def normalize_record(record: dict) -> dict:
    """
    Normalizes a dictionary record to ensure it contains all fields
    from the DB_SCHEMA.

    Args:
        record: A dictionary with partial data.

    Returns:
        A new dictionary containing all schema fields.
    """
    # Start with a copy of the default schema
    normalized = DB_SCHEMA.copy()
    # Overwrite the defaults with the actual values from the record
    normalized.update(record)
    return normalized


def process_text(cost_info, specified_type=None):
    cost_details = cost_info[COST_INFO]
    unique_id = cost_info["unique_id"]
    for item in cost_details:
        if item is None:
            logger.warning("None item in text skipping")
            continue
        output_tokens = item[COMPLETION_TOKENS]
        reasoning_tokes = item[REASONING_TOKENS]
        cached_input_tokens = item[CACHED_PROMPT_TEXT_TOKENS]
        input_tokens = item[PROMPT_TEXT_TOKENS]
        details = item[DETAILS]
        model = item[MODEL]
        real_input_tokens = input_tokens - cached_input_tokens
        assert real_input_tokens >= 0
        real_output_tokens = reasoning_tokes + output_tokens

        cost_input = real_input_tokens * text_table_cost[model]["input"]
        cost_cached_input = cached_input_tokens * text_table_cost[model]["input_cached"]
        cost_output = real_output_tokens * text_table_cost[model]["output"]

        total = cost_input + cost_cached_input + cost_output
        yield normalize_record(
            {
                "cost_input": cost_input,
                "cost_output": cost_output,
                "cost_cached_input": cost_cached_input,
                "total_cost": total,
                "details": details,
                "unique_id": unique_id,
                "model": model,
                COST_TYPE: specified_type
                or TEXT_PROCESSING,
            }
        )

