from typing import Literal, Optional, Any
import math

import instructor
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

class MedicalMCQResponse(BaseModel):
    """Response model for medical multiple choice questions"""
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


MEDICAL_BENCHMARK_SYSTEM_PROMPT = """You are a medical expert AI assistant tasked with answering multiple choice medical questions.

Instructions:
- Read the question carefully
- Analyze each answer option thoroughly
- Select the single best answer from the available choices (a, b, c, d, or e)
- Provide clear reasoning for your choice
- Be precise and evidence-based in your responses
- If multiple answers seem correct, choose the most accurate or comprehensive one"""

def get_single_answer_benchmark_vllm(
    model_name: str,
    question: str,
    options: dict[str, str],
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_logprobs: int = 5,
):
    """
    Get a single answer for a medical benchmark question using vLLM.

    Args:
        model_name: The model name (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        question: The medical question text
        options: Dictionary mapping choice letters to option text (e.g., {"a": "...", "b": "...", ...})
        base_url: vLLM server URL (default: "http://localhost:8000/v1")
        api_key: API key (use "EMPTY" for vLLM without auth)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_logprobs: Number of top logprobs to return per token (default: 5)

    Returns:
        dict: Response containing answer, usage info, and logprobs
    """
    # Create OpenAI client pointing to vLLM server
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Wrap with instructor for structured outputs using MD_JSON mode (better for non-OpenAI models)
    instructor_client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

    # Format options for the prompt
    options_text = "\n".join([f"{key.upper()}. {value}" for key, value in options.items()])
    user_prompt = f"{question}\n\nOptions:\n{options_text}"

    # Print Pydantic schema for debugging
    print("\n=== PYDANTIC MODEL SCHEMA ===")
    import json
    print(json.dumps(MedicalMCQResponse.model_json_schema(), indent=2))
    print("=" * 30 + "\n")

    # Enable instructor logging to see what it sends
    import logging
    logging.basicConfig(level=logging.DEBUG)
    instructor_logger = logging.getLogger("instructor")
    instructor_logger.setLevel(logging.DEBUG)

    try:
        (completion, raw) = instructor_client.chat.completions.create_with_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": MEDICAL_BENCHMARK_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=MedicalMCQResponse,
            max_retries=3,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        # Extract logprobs if available
        logprobs_data = None
        if raw.choices[0].logprobs:
            logprobs_data = {
                "content": raw.choices[0].logprobs.content,
            }

        # Return answer with usage info and logprobs
        result: dict[str, Any] = {
            "answer": completion.answer,
            "usage": {
                "prompt_tokens": raw.usage.prompt_tokens,
                "completion_tokens": raw.usage.completion_tokens,
                "total_tokens": raw.usage.total_tokens,
            }
        }

        if logprobs_data:
            result["logprobs"] = logprobs_data

        return result

    except Exception as e:
        raise RuntimeError(f"vLLM request failed: {str(e)}")


def extract_confidence_features(logprobs_data: dict[str, Any], answer: str) -> dict[str, float]:
    """
    Extract statistical features from logprobs for meta-learner.

    Features extracted:
    1. From reasoning tokens:
       - mean, median, std, min, max of logprobs
       - skewness and kurtosis (distribution shape)
    2. From answer token:
       - probability of the selected answer
       - probabilities of alternative answers (if available)
       - entropy of the answer distribution

    Args:
        logprobs_data: The logprobs dictionary from the response
        answer: The selected answer (e.g., "b")

    Returns:
        dict: Features for meta-learner
    """
    if not logprobs_data or 'content' not in logprobs_data:
        return {}

    content = logprobs_data['content']

    # Extract all logprobs (reasoning + answer)
    all_logprobs = [token.logprob for token in content]

    # Try to identify the answer token (looking for single letter a-e in quotes or alone)
    answer_token_idx = None
    answer_logprob = None
    answer_alternatives = {}

    for i, token_data in enumerate(content):
        token_text = token_data.token.strip().strip('"\'')
        if token_text.lower() == answer.lower():
            answer_token_idx = i
            answer_logprob = token_data.logprob

            # Extract alternatives from top_logprobs
            if token_data.top_logprobs:
                for alt in token_data.top_logprobs:
                    alt_text = alt.token.strip().strip('"\'').lower()
                    if alt_text in ['a', 'b', 'c', 'd', 'e']:
                        answer_alternatives[alt_text] = math.exp(alt.logprob)
            break

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

    # Entropy of answer distribution (uncertainty measure)
    if answer_alternatives:
        probs = list(answer_alternatives.values())
        # Normalize to ensure they sum to 1 (approximately)
        total = sum(probs)
        if total > 0:
            normalized_probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p + 1e-10) for p in normalized_probs if p > 0)
            features['answer_entropy'] = float(entropy)

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