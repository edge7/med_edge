"""
vLLM Native Structured Outputs - No Instructor
Uses vLLM's built-in response_format with JSON schema for full control.
Perfect for research papers where you need to know exactly what's sent to the model.
"""

from typing import Literal, Optional, Any
import math
import json
import logging

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from med_edge.llm_basic.prompts import MEDICAL_BENCHMARK_SYSTEM_PROMPT


# Disable noisy HTTP debug logs
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class MedicalMCQResponse(BaseModel):
    """Response model for medical multiple choice questions (5 options)"""
    answer: Literal["a", "b", "c", "d", "e"] = Field(
        description="The selected answer choice (a, b, c, d, or e). Must be exactly one of these lowercase letters."
    )

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Ensure answer is lowercase and one of the valid choices"""
        v = v.lower()
        if v not in ["a", "b", "c", "d", "e"]:
            raise ValueError(f"Answer must be one of a, b, c, d, e. Got: {v}")
        return v


def create_dynamic_mcq_response(valid_options: list[str], allow_abstain: bool = False, ask_prob: bool = False) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model for MCQ responses with specific valid options.

    Args:
        valid_options: List of valid answer options (e.g., ['a', 'b', 'c'] or ['a', 'b', 'c', 'd', 'e'])
        allow_abstain: If True, adds 'x' as a valid option for "I don't know" responses
        ask_prob: If True, adds a confidence field (1-100) for verbalized probability

    Returns:
        A Pydantic BaseModel class with the answer field restricted to the valid options
    """
    # Ensure options are lowercase
    valid_options = [opt.lower() for opt in valid_options]

    # Add abstention option if enabled
    if allow_abstain:
        valid_options = valid_options + ['x']

    # Create description string
    options_str = ", ".join(valid_options)
    if allow_abstain:
        description = f"The selected answer choice ({options_str}). Use 'x' if you are genuinely uncertain."
    else:
        description = f"The selected answer choice ({options_str}). Must be exactly one of these lowercase letters."

    # Create Literal type from options
    from typing import get_args
    literal_type = Literal[tuple(valid_options)]  # type: ignore

    # Dynamically create the model based on whether we need confidence
    if ask_prob:
        class DynamicMedicalMCQResponseWithConfidence(BaseModel):
            """Dynamically generated response model for medical multiple choice questions with confidence"""
            answer: literal_type = Field(description=description)  # type: ignore
            confidence: int = Field(
                description="Your confidence score from 1 to 100. 1 means very uncertain, 100 means absolutely certain.",
                ge=1,
                le=100
            )

            @field_validator("answer")
            @classmethod
            def validate_answer(cls, v: str) -> str:
                """Ensure answer is lowercase and one of the valid choices"""
                v = v.lower()
                if v not in valid_options:
                    raise ValueError(f"Answer must be one of {options_str}. Got: {v}")
                return v

        return DynamicMedicalMCQResponseWithConfidence
    else:
        class DynamicMedicalMCQResponse(BaseModel):
            """Dynamically generated response model for medical multiple choice questions"""
            answer: literal_type = Field(description=description)  # type: ignore

            @field_validator("answer")
            @classmethod
            def validate_answer(cls, v: str) -> str:
                """Ensure answer is lowercase and one of the valid choices"""
                v = v.lower()
                if v not in valid_options:
                    raise ValueError(f"Answer must be one of {options_str}. Got: {v}")
                return v

        return DynamicMedicalMCQResponse


def get_single_answer_vllm_native(
    model_name: str,
    question: str,
    options: dict[str, str],
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 32768,  # INCREASED: Prevent truncation for long reasoning (was 8192)
    top_logprobs: int = 20,  # Increased to capture all a-e alternatives + JSON tokens
    reasoning_effort: Optional[str] = None,  # "low", "mid", "high" for reasoning models
    verbose: bool = False,
    allow_abstain: bool = False,  # If True, allows "x" as "I don't know" response
    ask_prob: bool = False,  # If True, asks for verbalized confidence score (1-100)
):
    """
    Get a single answer using vLLM's native structured outputs (no instructor).

    This function gives you FULL CONTROL over what's sent to the model.
    No hidden prompt modifications - what you see is what you get.

    Args:
        model_name: The model name (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        question: The medical question text
        options: Dictionary mapping choice letters to option text (e.g., {"a": "...", "b": "...", ...})
        base_url: vLLM server URL (default: "http://localhost:8000/v1")
        api_key: API key (use "EMPTY" for vLLM without auth)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_logprobs: Number of top logprobs to return per token (default: 5)
        verbose: If True, prints the exact request being sent
        allow_abstain: If True, allows the model to respond with "x" meaning "I don't know"
        ask_prob: If True, asks for a verbalized confidence score (1-100) with the answer

    Returns:
        dict: Response containing answer, usage info, logprobs, and optionally verbalized_confidence
    """
    # Create OpenAI client pointing to vLLM server
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Format options for the prompt
    options_text = "\n".join([f"{key.upper()}. {value}" for key, value in options.items()])
    user_prompt = f"{question}\n\nOptions:\n{options_text}"

    # Build system prompt (with optional abstention instruction and confidence request)
    system_prompt = MEDICAL_BENCHMARK_SYSTEM_PROMPT
    if allow_abstain:
        system_prompt += "\n- If you are genuinely uncertain and cannot determine the best answer, you may respond with 'x' to indicate \"I don't know\" rather than guessing."
    if ask_prob:
        system_prompt += "\n- Along with your answer, provide a confidence score from 1 to 100 indicating how certain you are about your answer. 1 means very uncertain (essentially guessing), 100 means absolutely certain."

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Get JSON schema from Pydantic model
    # Use dynamic schema if options don't match the default 5 options (a-e) OR if abstention/ask_prob is enabled
    valid_options = sorted(options.keys())
    if set(valid_options) == {'a', 'b', 'c', 'd', 'e'} and not allow_abstain and not ask_prob:
        # Standard 5 options without abstention or confidence, use default model
        response_model = MedicalMCQResponse
        json_schema = response_model.model_json_schema()
    else:
        # Non-standard options OR abstention/ask_prob enabled, use dynamic model
        response_model = create_dynamic_mcq_response(valid_options, allow_abstain=allow_abstain, ask_prob=ask_prob)
        json_schema = response_model.model_json_schema()

    if verbose:
        print("\n" + "="*60)
        print("EXACT REQUEST BEING SENT TO vLLM")
        print("="*60)
        print(f"\nModel: {model_name}")
        print(f"\nMessages:")
        for i, msg in enumerate(messages, 1):
            print(f"\n--- Message {i} ({msg['role']}) ---")
            print(msg['content'])
        print(f"\nJSON Schema:")
        print(json.dumps(json_schema, indent=2))
        print(f"\nSampling params:")
        print(f"  temperature: {temperature}")
        print(f"  max_tokens: {max_tokens}")
        print(f"  top_logprobs: {top_logprobs}")
        if reasoning_effort:
            print(f"  reasoning_effort: {reasoning_effort}")
        print("="*60 + "\n")

    try:
        # Prepare extra body params for reasoning models
        extra_body = {}
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort

        # Use vLLM's native response_format (OpenAI compatible)
        # This sends ONLY what you specify - no hidden modifications
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "medical_mcq_response",
                    "schema": json_schema,
                    "strict": True,
                }
            },
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
            extra_body=extra_body if extra_body else None,
        )

        # Parse the JSON response
        content = response.choices[0].message.content

        if verbose:
            print(f"\n📝 FINAL ANSWER: {content}")

        # STRICT VALIDATION: content must not be empty
        if not content or content.strip() == "":
            raise RuntimeError(f"Model returned empty content. This should never happen!")

        parsed = json.loads(content)

        # Validate with Pydantic (use the appropriate model based on options)
        validated = response_model(**parsed)

        # STRICT VALIDATION: logprobs MUST be present
        if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
            raise RuntimeError(
                "Logprobs are REQUIRED but were not returned by the model. "
                "Ensure the model supports logprobs and the API is configured correctly."
            )

        logprobs_data = {
            "content": response.choices[0].logprobs.content,
        }

        # STRICT VALIDATION: reasoning_content MUST be present for reasoning models
        if not hasattr(response.choices[0].message, 'reasoning_content'):
            raise RuntimeError(
                "Reasoning content attribute is missing. "
                "Ensure you're using a reasoning model (DeepSeek R1, GPT-o1, etc.) "
                "or set reasoning_effort parameter."
            )

        reasoning_content = response.choices[0].message.reasoning_content

        if reasoning_content is None or reasoning_content == "":
            raise RuntimeError(
                "Reasoning content is REQUIRED but was empty or None. "
                "Ensure the model is generating reasoning (check reasoning_effort parameter)."
            )

        if verbose:
            print(f"\n🧠 REASONING DETECTED: {len(reasoning_content)} characters")
            print(f"First 200 chars: {reasoning_content[:200]}...")

        # Return answer with usage info, logprobs, reasoning, and finish_reason (ALL REQUIRED)
        result: dict[str, Any] = {
            "answer": validated.answer,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "logprobs": logprobs_data,
            "reasoning_content": reasoning_content,
            "finish_reason": response.choices[0].finish_reason,  # Track truncation!
        }

        # Add verbalized confidence if ask_prob was enabled
        if ask_prob and hasattr(validated, 'confidence'):
            result["verbalized_confidence"] = validated.confidence

        return result

    except Exception as e:
        raise RuntimeError(f"vLLM native request failed: {str(e)}")


def extract_confidence_features(logprobs_data: dict[str, Any], answer: str, valid_options: list[str] | None = None) -> dict[str, float]:
    """
    Extract statistical features from logprobs for meta-learner.

    Features extracted:
    1. From reasoning tokens:
       - mean, median, std, min, max of logprobs
       - skewness and kurtosis (distribution shape)
       - perplexity (fluidity of reasoning)
       - length and structure (token count, uniqueness, organization)
    2. From answer token:
       - probability of the selected answer
       - probabilities of alternative answers (if available)
       - entropy of the answer distribution
       - answer margin (top - second best option)

    Args:
        logprobs_data: The logprobs dictionary from the response
        answer: The selected answer (e.g., "b")
        valid_options: List of valid answer options (e.g., ['a', 'b', 'c', 'd']).
                      If None, defaults to all 5 options ['a', 'b', 'c', 'd', 'e']

    Returns:
        dict: Features for meta-learner
    """
    if not logprobs_data or 'content' not in logprobs_data:
        return {}

    # Default to all 5 options if not specified
    if valid_options is None:
        valid_options = ['a', 'b', 'c', 'd', 'e']

    content = logprobs_data['content']

    # Extract all logprobs (reasoning + answer)
    all_logprobs = [token.logprob for token in content]

    # Try to identify the answer token (looking for single letter in valid_options)
    # IMPORTANT: Search BACKWARDS from the end to find the LAST occurrence
    # This ensures we capture the actual JSON answer field, not mentions in reasoning
    answer_token_idx = None
    answer_logprob = None
    answer_alternatives = {}

    # Iterate backwards through content to find the last occurrence
    for i in range(len(content) - 1, -1, -1):
        token_data = content[i]
        token_text = token_data.token.strip().strip('"\'')
        if token_text.lower() == answer.lower():
            answer_token_idx = i
            answer_logprob = token_data.logprob

            # Extract alternatives from top_logprobs (only valid options)
            if token_data.top_logprobs:
                for alt in token_data.top_logprobs:
                    alt_text = alt.token.strip().strip('"\'').lower()
                    if alt_text in valid_options:
                        answer_alternatives[alt_text] = math.exp(alt.logprob)

                # Check if we found ALL valid options in top_logprobs
                missing_letters = set(valid_options) - set(answer_alternatives.keys())
                if missing_letters:
                    logger.warning(
                        f"⚠️  Missing {len(missing_letters)} valid options in top_logprobs: {sorted(missing_letters)}. "
                        f"Only found: {sorted(answer_alternatives.keys())} out of {sorted(valid_options)}. "
                        f"Consider increasing top_logprobs parameter."
                    )
            break  # Found the last occurrence, stop searching

    # Separate reasoning logprobs (exclude answer and special tokens after answer)
    reasoning_logprobs = []
    if answer_token_idx is not None:
        reasoning_logprobs = all_logprobs[:answer_token_idx]
    else:
        # If we can't find the answer token, use all logprobs
        reasoning_logprobs = all_logprobs

    # Calculate statistics on reasoning logprobs
    features = {}

    if reasoning_logprobs:
        reasoning_array = np.array(reasoning_logprobs)

        # Basic statistics
        features['reasoning_mean'] = float(np.mean(reasoning_array))
        features['reasoning_median'] = float(np.median(reasoning_array))
        features['reasoning_std'] = float(np.std(reasoning_array))
        features['reasoning_min'] = float(np.min(reasoning_array))
        features['reasoning_max'] = float(np.max(reasoning_array))

        # Distribution shape
        if len(reasoning_array) > 2:
            # Skewness: measure of asymmetry
            mean = np.mean(reasoning_array)
            std = np.std(reasoning_array)
            if std > 0:
                features['reasoning_skewness'] = float(np.mean(((reasoning_array - mean) / std) ** 3))
                # Kurtosis: measure of tail heaviness
                features['reasoning_kurtosis'] = float(np.mean(((reasoning_array - mean) / std) ** 4) - 3)

        # Trend: is the model getting more/less confident?
        if len(reasoning_array) > 10:
            first_half = np.mean(reasoning_array[:len(reasoning_array)//2])
            second_half = np.mean(reasoning_array[len(reasoning_array)//2:])
            features['reasoning_trend'] = float(second_half - first_half)

        # ============================================================
        # NEW FEATURE 2: Reasoning Perplexity (Fluidity)
        # ============================================================
        # Perplexity = exp(-mean_logprob)
        # Measures how "surprised" the model is by its own reasoning
        # Low perplexity = smooth reasoning, high perplexity = struggling
        features['reasoning_perplexity'] = float(np.exp(-features['reasoning_mean']))

        # ============================================================
        # NEW FEATURE 3: Reasoning Length & Structure (Qualitative Effort)
        # ============================================================
        # Extract reasoning tokens (before answer token)
        reasoning_tokens = []
        if answer_token_idx is not None:
            reasoning_tokens = [token_data.token for token_data in content[:answer_token_idx]]
        else:
            reasoning_tokens = [token_data.token for token_data in content]

        if reasoning_tokens:
            # Length: total number of reasoning tokens
            features['reasoning_length'] = len(reasoning_tokens)

            # Unique token ratio: measures variety (high = articulated, low = loops/repetition)
            unique_tokens = set(reasoning_tokens)
            features['unique_token_ratio'] = float(len(unique_tokens) / len(reasoning_tokens))

            # Newline frequency: measures structure/organization (paragraphs, lists)
            # Count newlines in token text
            newline_count = sum(1 for token in reasoning_tokens if '\n' in token)
            features['newline_frequency'] = float(newline_count / len(reasoning_tokens))
        else:
            features['reasoning_length'] = 0
            features['unique_token_ratio'] = 0.0
            features['newline_frequency'] = 0.0

    # Answer token probability (convert logprob to probability)
    if answer_logprob is not None:
        features['answer_probability'] = float(math.exp(answer_logprob))
        features['answer_logprob'] = float(answer_logprob)

    # Alternative probabilities
    for letter in ['a', 'b', 'c', 'd', 'e']:
        if letter in answer_alternatives:
            features[f'prob_{letter}'] = answer_alternatives[letter]
        else:
            features[f'prob_{letter}'] = 0.0

    # Initialize default values for entropy and margin (prevent missing values)
    features['answer_entropy'] = 0.0  # Default: no uncertainty when no alternatives
    features['answer_margin'] = 0.0   # Default: no margin when no alternatives

    # Entropy of answer distribution (uncertainty measure)
    if answer_alternatives:
        probs = list(answer_alternatives.values())
        # Normalize to ensure they sum to 1 (approximately)
        total = sum(probs)
        if total > 0:
            normalized_probs = [p / total for p in probs]

            # Calculate entropy correctly (FIXED BUG: was adding 1e-10 inside log)
            # Filter out near-zero probabilities BEFORE log to avoid issues
            # Entropy formula: H = -sum(p * log(p)) for p > 0
            entropy = 0.0
            for p in normalized_probs:
                if p > 1e-10:  # Only include non-negligible probabilities
                    entropy -= p * math.log(p)  # No +1e-10 inside log!

            features['answer_entropy'] = float(entropy)

        # ============================================================
        # NEW FEATURE 1: Answer Margin (Confidence Gap)
        # ============================================================
        # Margin = P_top - P_second
        # Measures the gap between the best and second-best answer
        # High margin = confident choice, low margin = coin flip / indecision
        sorted_probs = sorted(probs, reverse=True)
        if len(sorted_probs) >= 2:
            features['answer_margin'] = float(sorted_probs[0] - sorted_probs[1])
        elif len(sorted_probs) == 1:
            # Only one alternative available, margin is the probability itself
            features['answer_margin'] = float(sorted_probs[0])

    return features


def test_vllm_connection(
    model_name: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
):
    """
    Test connection to vLLM server with a simple request.

    Args:
        model_name: The model name loaded in vLLM
        base_url: vLLM server URL
        api_key: API key (use "EMPTY" for vLLM without auth)

    Returns:
        dict: Test results
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'Connection successful' if you can read this."}
            ],
            max_tokens=50,
            temperature=0.0,
        )

        return {
            "success": True,
            "model": model_name,
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model_name,
        }


def serialize_logprobs(logprobs_data, answer, reasoning_content):
    """
    Convert logprobs data to JSON-serializable format.
    The OpenAI response objects need to be converted to dicts.
    Bytes are excluded to save space.

    Uses JSON boundary detection to accurately separate reasoning tokens from JSON structure.

    Args:
        logprobs_data: Raw logprobs data from the model
        answer: The predicted answer (e.g., "b")
        reasoning_content: The reasoning text (not used in boundary detection but kept for compatibility)

    Returns:
        dict with:
            - content: array of all tokens with logprobs and accurate token_type
            - answer_token_index: position of the answer token
            - json_start_index: position where JSON starts
            - reasoning_end_index: position where reasoning ends
    """
    if not logprobs_data or 'content' not in logprobs_data:
        return None

    content = logprobs_data['content']

    # Find where reasoning ends by finding where JSON starts
    # JSON always starts with "{" token before the answer
    # This is much more reliable than text matching!
    reasoning_end_idx = None
    match_strategy = 'json_boundary'

    # Find answer token position (search backwards)
    answer_token_idx = None
    for i in range(len(content) - 1, -1, -1):
        token_text = content[i].token.strip().strip('"\'').lower()
        if token_text == answer.lower():
            answer_token_idx = i
            break

    # STRICT ASSERTIONS: Validate critical indices were found
    assert answer_token_idx is not None, \
        f"ASSERTION FAILED: Could not find answer token '{answer}' in logprobs content!"

    # Find JSON start by searching backwards from answer for "{" token
    # Safety: Don't search more than 50 tokens back (JSON structure should be small)
    MAX_JSON_SEARCH_DISTANCE = 50
    json_start_idx = None

    for i in range(answer_token_idx - 1, max(0, answer_token_idx - MAX_JSON_SEARCH_DISTANCE), -1):
        if content[i].token.startswith('{'):
            json_start_idx = i
            break

    # Sanity check: Did we find JSON start?
    if json_start_idx is None:
        logger.warning(f"⚠️  Could not find '{{' token within {MAX_JSON_SEARCH_DISTANCE} tokens before answer! "
                       f"Answer at {answer_token_idx}, searched back to {max(0, answer_token_idx - MAX_JSON_SEARCH_DISTANCE)}")
        # Fallback: assume JSON starts 10 tokens before answer (common pattern)
        json_start_idx = max(0, answer_token_idx - 10)
        logger.warning(f"⚠️  Using fallback: json_start_idx = {json_start_idx}")

    # Sanity check: Is JSON structure too large?
    json_distance = answer_token_idx - json_start_idx
    if json_distance > 20:
        logger.warning(f"⚠️  Large JSON structure detected! Distance from '{{' to answer: {json_distance} tokens. "
                       f"This might indicate reasoning contains '{{' characters.")

    # Reasoning ends just before JSON starts
    reasoning_end_idx = json_start_idx - 1

    # Sanity check: Do we have reasonable number of reasoning tokens?
    if reasoning_end_idx < 10:
        logger.warning(f"⚠️  Very few reasoning tokens detected: {reasoning_end_idx + 1}. "
                       f"This might indicate a problem with token structure.")

    # reasoning_end_idx should always be set now (via JSON boundary detection)

    assert reasoning_end_idx < answer_token_idx, \
        f"ASSERTION FAILED: Reasoning end ({reasoning_end_idx}) should be before answer ({answer_token_idx})!"

    # Serialize all tokens with accurate token_type
    serialized_content = []
    for idx, token_data in enumerate(content):
        # Determine token type based on position
        if reasoning_end_idx is not None and idx <= reasoning_end_idx:
            token_type = 'reasoning'
        elif answer_token_idx is not None and idx < answer_token_idx:
            token_type = 'json_structure'
        else:
            token_type = 'answer'

        token_dict = {
            'token': token_data.token,
            'logprob': token_data.logprob,
            'token_type': token_type
        }

        # Serialize top_logprobs if present
        if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
            # STRICT ASSERTION: All tokens must have exactly 20 top_logprobs
            assert len(token_data.top_logprobs) == 20, \
                f"ASSERTION FAILED: Token {idx} has {len(token_data.top_logprobs)} top_logprobs, expected 20!"

            token_dict['top_logprobs'] = [
                {
                    'token': alt.token,
                    'logprob': alt.logprob,
                }
                for alt in token_data.top_logprobs
            ]
        else:
            # STRICT ASSERTION: top_logprobs must exist
            raise AssertionError(f"ASSERTION FAILED: Token {idx} is missing top_logprobs!")

        serialized_content.append(token_dict)

    # STRICT ASSERTION: Must have reasoning tokens
    num_reasoning = reasoning_end_idx + 1
    assert num_reasoning > 0, "ASSERTION FAILED: No reasoning tokens found!"

    # STRICT ASSERTION: Total should match
    num_json_structure = answer_token_idx - reasoning_end_idx - 1
    num_answer = len(content) - answer_token_idx
    total_computed = num_reasoning + num_json_structure + num_answer
    assert total_computed == len(content), \
        f"ASSERTION FAILED: Token count mismatch! {total_computed} != {len(content)}"

    return {
        'content': serialized_content,
        'answer_token_index': answer_token_idx,
        'json_start_index': json_start_idx,  # Where "{" was found
        'reasoning_end_index': reasoning_end_idx,
        'reasoning_match_strategy': match_strategy,  # Track for transparency!
        'num_reasoning_tokens': num_reasoning,
        'num_json_structure_tokens': num_json_structure,
        'num_answer_tokens': num_answer,
        'num_total_tokens': len(content)
    }