"""
Test script to inspect reasoning_tokens and see if JSON structure tokens are included.
This will show us exactly what's in reasoning_tokens before the answer.
"""

from llm_basic.vllm_native_request import get_single_answer_vllm_native
from loguru import logger
import math

def main():
    # Your server configuration
    BASE_URL = "http://192.222.54.128:8000/v1"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # Simple test question
    question = "A 35-year-old man presents with fever and cough. What is the most likely diagnosis?"
    options = {
        "a": "Pneumonia",
        "b": "Common cold",
        "c": "Tuberculosis",
        "d": "Lung cancer",
        "e": "Asthma"
    }

    logger.info(f"Testing reasoning_tokens extraction with {MODEL_NAME}")

    # Get response
    response = get_single_answer_vllm_native(
        model_name=MODEL_NAME,
        question=question,
        options=options,
        base_url=BASE_URL,
        temperature=0.0,
        verbose=True,  # Debug mode to see what's returned
    )

    logger.success(f"✓ Answer: {response['answer']}")

    if 'logprobs' in response and response['logprobs']:
        content = response['logprobs']['content']
        answer = response['answer']

        # Find answer token (backward search like our code does)
        answer_token_idx = None
        for i in range(len(content) - 1, -1, -1):
            token_data = content[i]
            token_text = token_data.token.strip().strip('"\'')
            if token_text.lower() == answer.lower():
                answer_token_idx = i
                break

        if answer_token_idx is not None:
            logger.info(f"\n{'='*70}")
            logger.info(f"ANSWER TOKEN found at index {answer_token_idx}")
            logger.info(f"{'='*70}")

            # Extract reasoning_tokens exactly like the code does
            reasoning_tokens = [token_data.token for token_data in content[:answer_token_idx]]

            logger.info(f"\nTotal reasoning tokens: {len(reasoning_tokens)}")

            # Show LAST 20 reasoning tokens (right before answer)
            logger.info(f"\n{'='*70}")
            logger.info(f"LAST 20 REASONING TOKENS (right before answer)")
            logger.info(f"{'='*70}")

            start_idx = max(0, len(reasoning_tokens) - 20)
            for i in range(start_idx, len(reasoning_tokens)):
                token = reasoning_tokens[i]
                # Highlight JSON-like tokens
                is_json = token in ['{', '}', '"', ':', ',', 'answer', '{"', '":']
                marker = " ⚠️ JSON TOKEN" if is_json else ""
                logger.info(f"  [{i}] '{token}'{marker}")

            # Show what comes AFTER (the answer and closing JSON)
            logger.info(f"\n{'='*70}")
            logger.info(f"TOKENS AFTER ANSWER (JSON structure)")
            logger.info(f"{'='*70}")

            for i in range(answer_token_idx, min(answer_token_idx + 5, len(content))):
                token = content[i].token
                marker = " ← ANSWER" if i == answer_token_idx else ""
                logger.info(f"  [{i}] '{token}'{marker}")

            # Analyze reasoning_tokens content
            logger.info(f"\n{'='*70}")
            logger.info(f"ANALYSIS: JSON Tokens in reasoning_tokens")
            logger.info(f"{'='*70}")

            json_tokens = ['{', '}', '"', ':', ',', 'answer', '{"', '":']
            json_found = [t for t in reasoning_tokens if t in json_tokens]

            logger.info(f"JSON-like tokens found: {len(json_found)}")
            if json_found:
                logger.warning("⚠️  WARNING: JSON structure tokens found in reasoning_tokens!")
                logger.info(f"Tokens: {json_found[-10:]}")  # Show last 10
                logger.info(f"\nThese tokens contaminate:")
                logger.info(f"  - reasoning_length: includes JSON structure tokens")
                logger.info(f"  - unique_token_ratio: JSON tokens reduce uniqueness")

                # Calculate impact
                without_json = len([t for t in reasoning_tokens if t not in json_tokens])
                impact = len(reasoning_tokens) - without_json
                logger.info(f"\nImpact:")
                logger.info(f"  Total reasoning_tokens: {len(reasoning_tokens)}")
                logger.info(f"  Without JSON tokens: {without_json}")
                logger.info(f"  Contamination: {impact} tokens ({impact/len(reasoning_tokens)*100:.1f}%)")
            else:
                logger.success("✓ No JSON tokens found in reasoning_tokens!")

            # Show unique token ratio calculation
            unique_tokens = set(reasoning_tokens)
            unique_ratio = len(unique_tokens) / len(reasoning_tokens)
            logger.info(f"\n{'='*70}")
            logger.info(f"UNIQUE TOKEN RATIO")
            logger.info(f"{'='*70}")
            logger.info(f"Total tokens: {len(reasoning_tokens)}")
            logger.info(f"Unique tokens: {len(unique_tokens)}")
            logger.info(f"Ratio: {unique_ratio:.4f}")

            # Calculate without JSON contamination
            if json_found:
                tokens_no_json = [t for t in reasoning_tokens if t not in json_tokens]
                unique_no_json = set(tokens_no_json)
                ratio_no_json = len(unique_no_json) / len(tokens_no_json) if tokens_no_json else 0
                logger.info(f"\nWithout JSON contamination:")
                logger.info(f"  Total tokens: {len(tokens_no_json)}")
                logger.info(f"  Unique tokens: {len(unique_no_json)}")
                logger.info(f"  Ratio: {ratio_no_json:.4f}")
                logger.info(f"  Difference: {abs(unique_ratio - ratio_no_json):.4f}")

    else:
        logger.error("No logprobs available!")

if __name__ == "__main__":
    main()