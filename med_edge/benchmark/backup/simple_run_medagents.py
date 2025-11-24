"""
Simple benchmark script for medagents-benchmark - run directly from IDE.
Edit the config section below and execute.
"""

import json
import gzip
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

from med_edge.llm_basic.vllm_native_request import (
    get_single_answer_vllm_native,
    extract_confidence_features,
)


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
# All configs to run (excluding MedQA - use simple_run.py for that)
ALL_CONFIGS = [
    "AfrimedQA", "MMLU", "MMLU-Pro", "MedBullets", "MedExQA",
    "MedMCQA", "MedXpertQA-R", "MedXpertQA-U", "PubMedQA"
]

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
BASE_URL = "http://192.222.53.56:8000/v1"
LIMIT = None  # Number of samples to test per config (None = all)
TEMPERATURE = 0.6  # 1.0 for OPENAI; 0.6 DeepSeek
MAX_TOKENS = 32768  # Max tokens for generation
REASONING_EFFORT = None # None, "low", "mid", or "high" (for GPT-OSS only)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_test"
NUM_THREADS = 2  # Number of concurrent threads (2 is safe for vLLM)
# ============================================================================


def parse_medagents_sample(sample, is_hard=False):
    """Parse a medagents-benchmark dataset sample."""
    question = sample.get('question', '')

    # Options are already in dict format: {'A': 'text', 'B': 'text', ...}
    # Convert to lowercase keys to match our system
    options_raw = sample.get('options', {})
    options = {k.lower(): v for k, v in options_raw.items()}

    # Ground truth is in answer_idx field (already a letter like 'B')
    ground_truth = sample.get('answer_idx', '').lower()

    # Sample ID from realidx
    sample_id = sample.get('realidx', None)

    # Meta info
    meta_info = sample.get('meta_info', '')

    return {
        'question': question,
        'options': options,
        'ground_truth': ground_truth,
        'sample_id': sample_id,
        'meta_info': meta_info,
        'is_hard': is_hard,
    }


# Import serialize_logprobs from simple_run.py
def serialize_logprobs(logprobs_data, answer, reasoning_content):
    """
    Convert logprobs data to JSON-serializable format.
    The OpenAI response objects need to be converted to dicts.
    Bytes are excluded to save space.

    Uses JSON boundary detection to separate reasoning tokens from JSON structure.

    Args:
        logprobs_data: Raw logprobs data from the model
        answer: The predicted answer (e.g., "b")
        reasoning_content: The reasoning text (not used anymore but kept for compatibility)

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
                verbose=False,
            )

            predicted_answer = response['answer']
            is_correct = (predicted_answer == ground_truth)

            # Run feature extraction for validation/assertions only (don't save features)
            if 'logprobs' in response and response['logprobs']:
                valid_options = sorted(options.keys())
                _ = extract_confidence_features(response['logprobs'], predicted_answer, valid_options)
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


def run_single_config(dataset_config):
    """Run benchmark for a single config with resume support."""
    # Only show essential startup info
    logger.info(f"🚀 Starting benchmark: {MODEL_NAME} on medagents-benchmark/{dataset_config}")
    if LIMIT:
        logger.info(f"📊 Limit: {LIMIT} samples")

    # Load dataset from medagents-benchmark
    logger.info(f"📥 Loading medagents-benchmark config: {dataset_config}")
    test_data = load_dataset('super-dainiu/medagents-benchmark', dataset_config, split='test')
    test_hard_data = load_dataset('super-dainiu/medagents-benchmark', dataset_config, split='test_hard')

    # Create set of hard question IDs
    hard_ids = set([q['realidx'] for q in test_hard_data])
    logger.info(f"📝 Loaded {len(test_data)} test questions ({len(hard_ids)} marked as hard)")

    # Limit if specified
    if LIMIT:
        test_data = test_data.select(range(min(LIMIT, len(test_data))))

    # Prepare output file (NO timestamp for resume support)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    model_safe = MODEL_NAME.replace('/', '_')
    base_filename = f"{model_safe}_medagents_{dataset_config}_test"
    json_file = output_path / f"{base_filename}.jsonl.gz"  # Gzip compressed

    # ============================================================================
    # RESUME LOGIC: Load existing results and skip completed questions
    # ============================================================================
    completed_sample_ids = set()
    raw_results = [None] * len(test_data)  # Pre-allocate to maintain order

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
                    if idx is not None and 0 <= idx < len(test_data):
                        raw_results[idx] = item
                logger.info(f"✅ Found {len(completed_sample_ids)} already completed questions - will skip these")
        except Exception as e:
            logger.warning(f"⚠️  Could not load existing results: {e}")
            logger.warning(f"   Will start from scratch")

    # Count how many questions need processing
    questions_to_process = sum(1 for sample in test_data if sample['realidx'] not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"✅ All questions already completed for {dataset_config}!")
        return raw_results

    logger.info(f"📝 Processing {questions_to_process}/{len(test_data)} questions with {NUM_THREADS} threads ({len(test_data) - questions_to_process} already done)")

    SAVE_EVERY = 30  # Save every 30 samples

    # Wrapper function that includes idx for ordering
    def process_sample(idx, sample):
        # Skip if already completed
        if sample['realidx'] in completed_sample_ids:
            return idx, None

        # Check if this is a hard question
        is_hard = sample['realidx'] in hard_ids

        parsed = parse_medagents_sample(sample, is_hard=is_hard)

        raw_data = run_inference(
            question=parsed['question'],
            options=parsed['options'],
            ground_truth=parsed['ground_truth'],
        )

        # Add metadata to raw data
        raw_data['dataset_name'] = 'medagents-benchmark'
        raw_data['dataset_config'] = dataset_config
        raw_data['split_name'] = 'test'
        raw_data['question_index'] = idx
        raw_data['sample_id'] = parsed['sample_id']
        raw_data['is_hard'] = parsed['is_hard']
        raw_data['question'] = parsed['question']
        raw_data['options'] = parsed['options']
        raw_data['meta_info'] = parsed['meta_info']

        return idx, raw_data

    # Use ThreadPoolExecutor for concurrent inference
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit only tasks that haven't been completed yet
        futures = {executor.submit(process_sample, idx, sample): idx
                   for idx, sample in enumerate(test_data)
                   if sample['realidx'] not in completed_sample_ids}

        # Process completed tasks with progress bar
        completed_count = 0
        # Set total to questions_to_process for accurate progress tracking
        # Initialize progress bar with number of already completed questions
        with tqdm(total=len(test_data),
                  desc=f"Running inference [{dataset_config}]",
                  unit=" questions",
                  initial=len(test_data) - questions_to_process) as pbar:
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

                                logger.info(f"💾 Checkpoint: {len(completed_raw)}/{len(test_data)}")
                            except Exception as save_err:
                                logger.error(f"⚠️  Checkpoint save failed (continuing): {save_err}")

                except Exception as e:
                    logger.error(f"Task {idx} failed: {str(e)}")
                    # Create error result
                    raw_results[idx] = {
                        'dataset_name': 'medagents-benchmark',
                        'dataset_config': dataset_config,
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

    # Calculate hard subset accuracy
    hard_results = [r for r in completed_results if r.get('is_hard', False)]
    if hard_results:
        hard_correct = sum(1 for r in hard_results if r.get('is_correct', False))
        hard_accuracy = hard_correct / len(hard_results)
        logger.success(f"\nAccuracy (all): {accuracy:.2%} ({correct_count}/{total_count})")
        logger.success(f"Accuracy (hard): {hard_accuracy:.2%} ({hard_correct}/{len(hard_results)})")
    else:
        logger.success(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{total_count})")

    logger.success(f"Results saved to: {json_file}")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dataset: medagents-benchmark/{dataset_config}")
    logger.info(f"Samples: {total_count}")
    logger.info(f"Accuracy: {accuracy:.2%}")

    return completed_results


def main():
    """Main function to loop through all configs with resume support."""
    logger.info("="*80)
    logger.info(f"🔬 Running ALL medagents-benchmark configs")
    logger.info(f"   Model: {MODEL_NAME}")
    logger.info(f"   Total configs: {len(ALL_CONFIGS)}")
    logger.info(f"   Configs: {', '.join(ALL_CONFIGS)}")
    logger.info("="*80)

    results_summary = {}

    for idx, config in enumerate(ALL_CONFIGS, 1):
        logger.info(f"\n\n{'='*80}")
        logger.info(f"📌 CONFIG {idx}/{len(ALL_CONFIGS)}: {config}")
        logger.info(f"{'='*80}\n")

        try:
            results = run_single_config(config)
            correct_count = sum(1 for r in results if r.get('is_correct', False))
            accuracy = correct_count / len(results) if len(results) > 0 else 0.0
            results_summary[config] = {
                'samples': len(results),
                'accuracy': accuracy,
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"❌ Failed to process config {config}: {str(e)}")
            results_summary[config] = {
                'samples': 0,
                'accuracy': 0.0,
                'status': 'failed',
                'error': str(e)
            }

    # Print final summary for all configs
    logger.info("\n\n" + "="*80)
    logger.info("🏁 FINAL SUMMARY - ALL CONFIGS")
    logger.info("="*80)
    for config, stats in results_summary.items():
        if stats['status'] == 'completed':
            logger.info(f"  {config:15s}: {stats['accuracy']:.2%} ({stats['samples']} samples)")
        else:
            logger.error(f"  {config:15s}: FAILED - {stats.get('error', 'Unknown error')}")
    logger.info("="*80)


if __name__ == "__main__":
    main()