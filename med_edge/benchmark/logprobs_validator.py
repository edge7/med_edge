"""
Strict validation for logprobs data integrity.
For research purposes - ensures no data loss during serialization.

Raises exceptions if any data is missing or corrupted.
"""

import math
from typing import Any


class LogprobsValidationError(Exception):
    """Raised when logprobs data fails validation."""
    pass


def validate_serialized_logprobs(serialized_data: dict, expected_answer: str) -> dict:
    """
    Strictly validate serialized logprobs data.

    This is CRITICAL for research - ensures:
    1. All tokens are present with logprobs
    2. All tokens have exactly 20 top_logprobs
    3. Answer token is present and identifiable
    4. Reasoning tokens are correctly separated from JSON structure
    5. Answer probability can be extracted

    Args:
        serialized_data: Output from serialize_logprobs()
        expected_answer: The predicted answer (e.g., 'b')

    Returns:
        dict with validation metrics

    Raises:
        LogprobsValidationError: If any validation check fails
    """
    if serialized_data is None:
        raise LogprobsValidationError("Serialized data is None!")

    # Check required top-level fields
    required_fields = [
        'content', 'answer_token_index', 'json_start_index',
        'reasoning_end_index', 'num_reasoning_tokens',
        'num_json_structure_tokens', 'num_answer_tokens', 'num_total_tokens'
    ]

    for field in required_fields:
        if field not in serialized_data:
            raise LogprobsValidationError(f"Missing required field: {field}")

    content = serialized_data['content']

    if not isinstance(content, list):
        raise LogprobsValidationError(f"content must be a list, got {type(content)}")

    if len(content) == 0:
        raise LogprobsValidationError("content is empty!")

    # Validate token count matches
    stated_total = serialized_data['num_total_tokens']
    actual_total = len(content)

    if stated_total != actual_total:
        raise LogprobsValidationError(
            f"Token count mismatch: stated {stated_total} but have {actual_total} tokens"
        )

    # Validate sum of token types equals total
    num_reasoning = serialized_data['num_reasoning_tokens']
    num_json = serialized_data['num_json_structure_tokens']
    num_answer = serialized_data['num_answer_tokens']
    computed_total = num_reasoning + num_json + num_answer

    if computed_total != actual_total:
        raise LogprobsValidationError(
            f"Token type sum mismatch: {num_reasoning} + {num_json} + {num_answer} = {computed_total} != {actual_total}"
        )

    # Validate we have reasoning tokens
    if num_reasoning == 0:
        raise LogprobsValidationError("No reasoning tokens found!")

    if num_reasoning < 10:
        raise LogprobsValidationError(
            f"Suspiciously few reasoning tokens: {num_reasoning}. Possible boundary detection error."
        )

    # Validate each token
    answer_token_found = False
    reasoning_count = 0
    json_count = 0
    answer_count = 0

    for idx, token_data in enumerate(content):
        # Check required token fields
        if 'token' not in token_data:
            raise LogprobsValidationError(f"Token {idx} missing 'token' field")

        if 'logprob' not in token_data:
            raise LogprobsValidationError(f"Token {idx} missing 'logprob' field")

        if 'token_type' not in token_data:
            raise LogprobsValidationError(f"Token {idx} missing 'token_type' field")

        if 'top_logprobs' not in token_data:
            raise LogprobsValidationError(f"Token {idx} missing 'top_logprobs' field")

        # Validate logprob is a number
        if not isinstance(token_data['logprob'], (int, float)):
            raise LogprobsValidationError(
                f"Token {idx} logprob must be numeric, got {type(token_data['logprob'])}"
            )

        # Validate logprob is in reasonable range (log probabilities should be <= 0)
        if token_data['logprob'] > 0.01:  # Allow tiny positive due to floating point
            raise LogprobsValidationError(
                f"Token {idx} logprob is suspiciously high: {token_data['logprob']} (should be <= 0)"
            )

        # CRITICAL: Validate exactly 20 top_logprobs
        top_logprobs = token_data['top_logprobs']
        if not isinstance(top_logprobs, list):
            raise LogprobsValidationError(
                f"Token {idx} top_logprobs must be a list, got {type(top_logprobs)}"
            )

        if len(top_logprobs) != 20:
            raise LogprobsValidationError(
                f"Token {idx} must have exactly 20 top_logprobs, got {len(top_logprobs)}"
            )

        # Validate each top_logprob entry
        for alt_idx, alt in enumerate(top_logprobs):
            if 'token' not in alt:
                raise LogprobsValidationError(
                    f"Token {idx} top_logprob {alt_idx} missing 'token' field"
                )

            if 'logprob' not in alt:
                raise LogprobsValidationError(
                    f"Token {idx} top_logprob {alt_idx} missing 'logprob' field"
                )

            if not isinstance(alt['logprob'], (int, float)):
                raise LogprobsValidationError(
                    f"Token {idx} top_logprob {alt_idx} logprob must be numeric"
                )

            if alt['logprob'] > 0.01:
                raise LogprobsValidationError(
                    f"Token {idx} top_logprob {alt_idx} logprob suspiciously high: {alt['logprob']}"
                )

        # Count token types
        token_type = token_data['token_type']
        if token_type == 'reasoning':
            reasoning_count += 1
        elif token_type == 'json_structure':
            json_count += 1
        elif token_type == 'answer':
            answer_count += 1
        else:
            raise LogprobsValidationError(
                f"Token {idx} has invalid token_type: {token_type}"
            )

        # Check if this is the answer token
        token_text = token_data['token'].strip().strip('"\'').lower()
        if idx == serialized_data['answer_token_index']:
            answer_token_found = True
            if token_text != expected_answer.lower():
                raise LogprobsValidationError(
                    f"Answer token at index {idx} is '{token_text}' but expected '{expected_answer}'"
                )

    # Validate answer token was found
    if not answer_token_found:
        raise LogprobsValidationError(
            f"Answer token '{expected_answer}' not found at expected index "
            f"{serialized_data['answer_token_index']}"
        )

    # Validate token type counts match
    if reasoning_count != num_reasoning:
        raise LogprobsValidationError(
            f"Reasoning token count mismatch: stated {num_reasoning}, found {reasoning_count}"
        )

    if json_count != num_json:
        raise LogprobsValidationError(
            f"JSON token count mismatch: stated {num_json}, found {json_count}"
        )

    if answer_count != num_answer:
        raise LogprobsValidationError(
            f"Answer token count mismatch: stated {num_answer}, found {answer_count}"
        )

    # Extract answer probability from top_logprobs
    answer_token_idx = serialized_data['answer_token_index']
    answer_token_data = content[answer_token_idx]
    answer_probability = math.exp(answer_token_data['logprob'])

    # Validate answer probability is reasonable
    if not (0.0 <= answer_probability <= 1.0):
        raise LogprobsValidationError(
            f"Answer probability {answer_probability} is not in [0, 1]"
        )

    # Return validation summary
    return {
        'valid': True,
        'total_tokens': actual_total,
        'reasoning_tokens': reasoning_count,
        'json_structure_tokens': json_count,
        'answer_tokens': answer_count,
        'answer_probability': answer_probability,
        'answer_token': answer_token_data['token'],
        'answer_logprob': answer_token_data['logprob'],
    }


def validate_benchmark_result(result: dict) -> bool:
    """
    Validate a complete benchmark result dict.

    Ensures all required fields are present for research reproducibility.

    Args:
        result: A result dict from benchmark run

    Returns:
        True if valid

    Raises:
        LogprobsValidationError: If validation fails
    """
    # Required fields for open-source models (with logprobs)
    required_fields = [
        'answer', 'ground_truth', 'is_correct', 'usage',
        'reasoning_content', 'finish_reason', 'temperature',
        'max_tokens', 'logprobs'
    ]

    for field in required_fields:
        if field not in result:
            raise LogprobsValidationError(f"Result missing required field: {field}")

    # Validate usage has token counts
    usage = result['usage']
    for field in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
        if field not in usage:
            raise LogprobsValidationError(f"Usage missing field: {field}")

        if not isinstance(usage[field], int) or usage[field] < 0:
            raise LogprobsValidationError(
                f"Usage {field} must be non-negative integer, got {usage[field]}"
            )

    # Validate logprobs
    if result['logprobs'] is None:
        raise LogprobsValidationError("Logprobs is None - required for research!")

    # Validate serialized logprobs structure
    validate_serialized_logprobs(result['logprobs'], result['answer'])

    return True


def extract_answer_probabilities(serialized_logprobs: dict, valid_options: list[str]) -> dict[str, float]:
    """
    Extract answer probabilities for all valid options from top_logprobs.

    This is critical for meta-learning and confidence calibration research.

    Args:
        serialized_logprobs: Output from serialize_logprobs()
        valid_options: List of valid answer options (e.g., ['a', 'b', 'c', 'd'])

    Returns:
        dict mapping option letter to probability

    Raises:
        LogprobsValidationError: If answer probabilities cannot be extracted
    """
    answer_token_idx = serialized_logprobs['answer_token_index']
    content = serialized_logprobs['content']
    answer_token = content[answer_token_idx]

    top_logprobs = answer_token['top_logprobs']

    # Extract probabilities for each valid option
    probabilities = {}

    for alt in top_logprobs:
        alt_text = alt['token'].strip().strip('"\'').lower()
        if alt_text in valid_options:
            probabilities[alt_text] = math.exp(alt['logprob'])

    # Warn if we're missing any valid options
    missing_options = set(valid_options) - set(probabilities.keys())
    if missing_options:
        raise LogprobsValidationError(
            f"Cannot extract probabilities for options {sorted(missing_options)}. "
            f"Only found {sorted(probabilities.keys())} in top 20 logprobs. "
            f"This means the model assigned very low probability to these options. "
            f"Consider increasing top_logprobs parameter or these options are truly unlikely."
        )

    return probabilities
