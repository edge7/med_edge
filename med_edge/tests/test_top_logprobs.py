"""
Test script to inspect top_logprobs content with top_logprobs=20
This will help us understand why we only get 1 alternative in 83% of cases.
"""

from llm_basic.vllm_native_request import get_single_answer_vllm_native, extract_confidence_features
from loguru import logger
import json

def main():
    # Your server configuration
    BASE_URL = "http://192.222.54.128:8000/v1"
    MODEL_NAME = "openai/gpt-oss-120b"  # Adjust if needed

    # Simple test question
    question = "A 35-year-old man presents with fever and cough. What is the most likely diagnosis?"
    options = {
        "a": "Pneumonia",
        "b": "Common cold",
        "c": "Tuberculosis",
        "d": "Lung cancer",
        "e": "Asthma"
    }

    logger.info(f"Testing with {MODEL_NAME} at {BASE_URL}")
    logger.info("Running with top_logprobs=20 (new default)")

    # Get response with verbose output
    response = get_single_answer_vllm_native(
        model_name=MODEL_NAME,
        question=question,
        options=options,
        base_url=BASE_URL,
        temperature=0.0,
        verbose=True,  # See full request
    )

    logger.success(f"\n✓ Answer: {response['answer']}")

    # Deep dive into logprobs
    if 'logprobs' in response and response['logprobs']:
        content = response['logprobs']['content']

        logger.info(f"\n{'='*60}")
        logger.info(f"LOGPROBS ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Total tokens: {len(content)}")

        # Find the answer token (backward search like our code does)
        answer = response['answer']
        answer_token_idx = None

        for i in range(len(content) - 1, -1, -1):
            token_data = content[i]
            token_text = token_data.token.strip().strip('"\'')
            if token_text.lower() == answer.lower():
                answer_token_idx = i
                break

        if answer_token_idx is not None:
            logger.info(f"\n🎯 ANSWER TOKEN FOUND at index {answer_token_idx}")
            logger.info(f"Token: '{content[answer_token_idx].token}'")
            logger.info(f"Logprob: {content[answer_token_idx].logprob:.4f}")
            logger.info(f"Probability: {math.exp(content[answer_token_idx].logprob):.4f}")

            # Show last 10 tokens (including answer)
            logger.info(f"\n📝 LAST 10 TOKENS (JSON structure):")
            start_idx = max(0, answer_token_idx - 5)
            end_idx = min(len(content), answer_token_idx + 5)

            for i in range(start_idx, end_idx):
                token_data = content[i]
                marker = " ← ANSWER" if i == answer_token_idx else ""
                logger.info(f"  [{i}] '{token_data.token}' (logprob: {token_data.logprob:.4f}){marker}")

            # Analyze top_logprobs for the answer token
            answer_token = content[answer_token_idx]
            if answer_token.top_logprobs:
                logger.info(f"\n🔍 TOP_LOGPROBS for answer token (total: {len(answer_token.top_logprobs)}):")

                letters_found = []
                json_tokens = []
                other_tokens = []

                for alt in answer_token.top_logprobs:
                    alt_text = alt.token.strip().strip('"\'').lower()
                    prob = math.exp(alt.logprob)

                    if alt_text in ['a', 'b', 'c', 'd', 'e']:
                        letters_found.append((alt_text, prob, alt.logprob))
                        logger.info(f"  ✓ LETTER '{alt_text}': prob={prob:.6f}, logprob={alt.logprob:.4f}")
                    elif alt.token in ['{', '}', '"', ':', ',']:
                        json_tokens.append((alt.token, prob, alt.logprob))
                        logger.info(f"    JSON '{alt.token}': prob={prob:.6f}, logprob={alt.logprob:.4f}")
                    else:
                        other_tokens.append((alt.token, prob, alt.logprob))
                        logger.info(f"    OTHER '{alt.token}': prob={prob:.6f}, logprob={alt.logprob:.4f}")

                logger.info(f"\n📊 SUMMARY:")
                logger.info(f"  Letters (a-e) found: {len(letters_found)}/5")
                logger.info(f"  JSON tokens: {len(json_tokens)}")
                logger.info(f"  Other tokens: {len(other_tokens)}")

                missing_letters = set(['a', 'b', 'c', 'd', 'e']) - set([l[0] for l in letters_found])
                if missing_letters:
                    logger.warning(f"  ⚠️  Missing letters: {missing_letters}")
                    logger.warning(f"     These letters are NOT in top {len(answer_token.top_logprobs)} alternatives!")
                else:
                    logger.success(f"  ✓ All 5 letters present in top_logprobs!")

        # Extract features
        logger.info(f"\n{'='*60}")
        logger.info(f"EXTRACTED FEATURES")
        logger.info(f"{'='*60}")

        features = extract_confidence_features(response['logprobs'], response['answer'])

        for key, value in sorted(features.items()):
            logger.info(f"  {key:25s}: {value}")

    else:
        logger.error("No logprobs available!")

if __name__ == "__main__":
    import math
    main()