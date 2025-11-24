"""
Simple benchmark script - run directly from IDE.
Edit the config section below and execute.
"""

import json
import gzip
import hashlib
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from med_edge.dataset_handler.hugging_face_handler import get_med_qa_dataset
from med_edge.llm_basic.vllm_native_request import (
    get_single_answer_vllm_native,
    extract_confidence_features,
)


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
MODEL_NAME = "Qwen/Qwen3-32B"
BASE_URL = "http://192.222.54.244:8000/v1"
SPLIT = "test"  # train, val, or test
LIMIT = None  # Number of samples to test (None = all)
TEMPERATURE = 0.6 # 1.0 for OPENAI; 0.6 for DeepSeek; 0.6 OLMO
MAX_TOKENS = 22768  # Max tokens for generation (must match vllm_native_request.py default)
REASONING_EFFORT = None  # None, "low", "mid", or "high" (for GPT-OSS only)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks"
NUM_THREADS = 3  # Number of concurrent threads (2 is safe for vLLM)
VERBOSE = False  # Set to True to verify exact prompts being sent (for research verification)
# ============================================================================


def parse_medqa_sample(sample, idx=None):
    """
    Parse a MedQA dataset sample.

    Args:
        sample: Dataset sample dict
        idx: Question index in the dataset (used as fallback for sample_id)
    """
    question = sample.get('question', '')

    # Parse options
    options = {}
    for opt in sample.get('options', []):
        key = opt.get('key', '').lower()
        value = opt.get('value', '')
        options[key] = value

    # Find ground truth
    ground_truth_value = sample.get('answer', '')
    ground_truth = None
    for key, value in options.items():
        if value == ground_truth_value:
            ground_truth = key
            break

    if ground_truth is None:
        ground_truth = ground_truth_value.lower()

    # CRITICAL: Use hash of question text as sample_id for 100% determinism
    # This ensures same question = same ID even if dataset order changes
    # Using MD5 hash truncated to 64-bit integer
    question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
    sample_id = int(question_hash[:16], 16)  # First 64 bits as int

    return {
        'question': question,
        'options': options,
        'ground_truth': ground_truth,
        'sample_id': sample_id,
        'question_index': idx,  # Keep original index for reference
        'meta_info': sample.get('meta_info', None),  # Keep for reference
    }


def run_inference(question, options, ground_truth, max_retries=2):
    """
    Run inference on a single question with retry logic.

    Returns:
        dict: Complete raw response data for JSON
    """
    import time

    for attempt in range(max_retries + 1):
        try:
            response = get_single_answer_vllm_native(
                model_name=MODEL_NAME,
                question=question,
                options=options,
                base_url=BASE_URL,
                api_key="EMPTY",
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                reasoning_effort=REASONING_EFFORT,
                verbose=VERBOSE,
            )

            predicted_answer = response['answer']
            is_correct = (predicted_answer == ground_truth)

            # Run feature extraction for validation/assertions only (don't save features)
            if 'logprobs' in response and response['logprobs']:
                _ = extract_confidence_features(response['logprobs'], predicted_answer)
            else:
                raise Exception("Logprobs must be available!")

            # Raw data for JSON (complete response + metadata)
            raw_data = {
                'answer': response['answer'],
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'usage': response['usage'],
                'reasoning_content': response.get('reasoning_content', ''),
                'finish_reason': response.get('finish_reason', None),  # Track if truncated!
                'temperature': TEMPERATURE,  # Sampling parameter for reproducibility
                'max_tokens': MAX_TOKENS,  # Max tokens used in generation
                'reasoning_effort': REASONING_EFFORT,  # Reasoning effort level
            }

            # Include raw logprobs if available
            if 'logprobs' in response and response['logprobs']:
                # Convert logprobs to serializable format (includes assertions)
                raw_data['logprobs'] = serialize_logprobs(
                    response['logprobs'],
                    predicted_answer,
                    response.get('reasoning_content', '')
                )

            return raw_data

        except Exception as e:
            # Extract just the error message, not the full stack trace
            error_msg = str(e).split('\n')[0][:200]  # First line, max 200 chars

            if attempt < max_retries:
                logger.warning(f"⚠️  Attempt {attempt + 1} failed: {error_msg}. Retrying in 2s...")
                time.sleep(2)
            else:
                logger.error(f"❌ All {max_retries + 1} attempts failed: {error_msg}")
                return {
                    'answer': None,
                    'ground_truth': ground_truth,
                    'is_correct': False,
                    'error': error_msg,
                }
    return None


def serialize_logprobs(logprobs_data, answer, reasoning_content):
    """
    Convert logprobs data to JSON-serializable format.
    The OpenAI response objects need to be converted to dicts.
    Bytes are excluded to save space.

    Uses reasoning_content to accurately separate reasoning tokens from JSON structure.

    Args:
        logprobs_data: Raw logprobs data from the model
        answer: The predicted answer (e.g., "b")
        reasoning_content: The reasoning text to identify where reasoning ends

    Returns:
        dict with:
            - content: array of all tokens with logprobs and accurate token_type
            - answer_token_index: position of the answer token
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


def main():
    # Only show essential startup info
    logger.info(f"🚀 Starting benchmark: {MODEL_NAME} on {SPLIT} split")
    if LIMIT:
        logger.info(f"📊 Limit: {LIMIT} samples")

    # Load dataset (suppress verbose logs)
    dataset = get_med_qa_dataset()

    if SPLIT == 'train':
        data = dataset.train
    elif SPLIT == 'val':
        data = dataset.val
    else:
        data = dataset.test

    # Limit if specified
    if LIMIT:
        data = data.select(range(min(LIMIT, len(data))))

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    model_safe = MODEL_NAME.replace('/', '_')
    base_filename = f"{model_safe}_{SPLIT}"
    json_file = output_path / f"{base_filename}.jsonl.gz"  # Gzip compressed

    # ============================================================================
    # RESUME LOGIC: Load existing results and skip completed questions
    # ============================================================================
    completed_sample_ids = set()
    raw_results = [None] * len(data)  # Pre-allocate to maintain order

    if json_file.exists():
        logger.info(f"📂 Found existing results file: {json_file}")
        try:
            with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                existing_raw = json.load(f)
                # Pre-populate raw results
                for item in existing_raw:
                    idx = item.get('question_index')
                    sample_id = item.get('sample_id')
                    if sample_id is not None:
                        completed_sample_ids.add(sample_id)
                    if idx is not None and 0 <= idx < len(data):
                        raw_results[idx] = item
                logger.info(f"✅ Found {len(completed_sample_ids)} already completed questions - will skip these")
        except Exception as e:
            logger.warning(f"⚠️  Could not load existing results: {e}")
            logger.warning(f"   Will start from scratch")

    # Helper function to extract sample_id consistently (matches parse_medqa_sample logic)
    def get_sample_id(sample):
        # CRITICAL: Use hash of question text for 100% determinism
        question = sample.get('question', '')
        question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        return int(question_hash[:16], 16)

    # Count how many questions need processing
    questions_to_process = sum(1 for sample in data if get_sample_id(sample) not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"✅ All questions already completed for {SPLIT}!")
        return raw_results

    logger.info(f"📝 Processing {questions_to_process}/{len(data)} questions with {NUM_THREADS} threads ({len(data) - questions_to_process} already done)")

    SAVE_EVERY = 150

    # Wrapper function that includes idx for ordering
    def process_sample(idx, sample):
        # Skip if already completed (use hash as sample_id)
        sample_id = get_sample_id(sample)
        if sample_id in completed_sample_ids:
            return idx, None

        parsed = parse_medqa_sample(sample, idx=idx)

        raw_data = run_inference(
            question=parsed['question'],
            options=parsed['options'],
            ground_truth=parsed['ground_truth'],
        )

        # Add metadata to raw data
        raw_data['dataset_name'] = 'medqa'
        raw_data['split_name'] = SPLIT
        raw_data['question_index'] = idx
        raw_data['sample_id'] = parsed['sample_id']
        raw_data['question'] = parsed['question']
        raw_data['options'] = parsed['options']
        raw_data['meta_info'] = parsed.get('meta_info', None)  # step1 or step2&3

        return idx, raw_data

    # Use ThreadPoolExecutor for concurrent inference
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit only tasks that haven't been completed yet (use hash as sample_id)
        futures = {executor.submit(process_sample, idx, sample): idx
                   for idx, sample in enumerate(data)
                   if get_sample_id(sample) not in completed_sample_ids}

        # Process completed tasks with progress bar
        completed_count = 0
        # Initialize progress bar at current completion level
        with tqdm(total=len(data),
                  desc=f"Running inference [{SPLIT}]",
                  unit=" questions",
                  initial=len(data) - questions_to_process) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_idx, raw_data = future.result()
                    # Only update if not None (not skipped)
                    if raw_data is not None:
                        raw_results[result_idx] = raw_data
                        completed_count += 1
                        pbar.update(1)

                        # Save checkpoint every SAVE_EVERY samples
                        if completed_count % SAVE_EVERY == 0:
                            try:
                                # Filter out None values (not yet completed)
                                completed_raw = [r for r in raw_results if r is not None]

                                # ATOMIC WRITE: Save to temp file first, then rename
                                json_temp = json_file.with_suffix('.jsonl.gz.tmp')
                                with gzip.open(json_temp, 'wt', encoding='utf-8', compresslevel=6) as f:
                                    json.dump(completed_raw, f)
                                json_temp.replace(json_file)  # Atomic rename

                                logger.info(f"💾 Checkpoint: {len(completed_raw)}/{len(data)}")
                            except Exception as save_err:
                                logger.error(f"⚠️  Checkpoint save failed (continuing): {save_err}")
                                # Continue processing even if save fails

                except Exception as e:
                    logger.error(f"Task {idx} failed: {str(e)}")
                    # Create error result
                    raw_results[idx] = {
                        'dataset_name': 'medqa',
                        'question_index': idx,
                        'answer': None,
                        'is_correct': False,
                        'error': str(e),
                    }
                    pbar.update(1)

    # Save final raw data as compressed JSON atomically
    logger.info("Saving final JSON")
    json_temp = json_file.with_suffix('.jsonl.gz.tmp')
    with gzip.open(json_temp, 'wt', encoding='utf-8', compresslevel=6) as f:
        json.dump(raw_results, f)
    json_temp.replace(json_file)  # Atomic rename

    # Calculate accuracy from raw_results
    completed_results = [r for r in raw_results if r is not None]
    correct_count = sum(1 for r in completed_results if r.get('is_correct', False))
    total_count = len(completed_results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    logger.success(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    logger.success(f"Results saved to: {json_file}")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Samples: {total_count}")
    logger.info(f"Accuracy: {accuracy:.2%}")

    return raw_results


if __name__ == "__main__":
    main()