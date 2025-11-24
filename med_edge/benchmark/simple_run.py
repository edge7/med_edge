"""
Clean benchmark script for MedQA using open-source models (vLLM).
Refactored to use shared utilities from benchmark_utils.py
"""

import hashlib
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from med_edge.dataset_handler.hugging_face_handler import get_med_qa_dataset
from med_edge.llm_basic.vllm_native_request import (
    get_single_answer_vllm_native,
    extract_confidence_features,
    serialize_logprobs,
)
from med_edge.benchmark.benchmark_utils import (
    parse_medqa_sample,
    atomic_save_json,
    setup_resume_logic,
    calculate_accuracy_summary,
    log_accuracy_summary,
    get_sample_id_from_medqa,
)


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
BASE_URL = "http://192.222.53.56:8000/v1"
SPLIT = "test"  # train, val, or test
LIMIT = None  # Number of samples to test (None = all)
TEMPERATURE = 0.6  # 1.0 for OPENAI; 0.6 DeepSeek
MAX_TOKENS = 32768  # Max tokens for generation
REASONING_EFFORT = None  # None, "low", "mid", or "high" (for GPT-OSS only)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_test"
NUM_THREADS = 4  # Number of concurrent threads (4 is safe for vLLM)
VERBOSE = False  # Set to True to see exact prompts
# ============================================================================


def run_inference(question, options, ground_truth, max_retries=2):
    """Run inference with retry logic."""
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

            # Run feature extraction for validation/assertions only
            if 'logprobs' in response and response['logprobs']:
                _ = extract_confidence_features(response['logprobs'], predicted_answer)
            else:
                raise Exception("Logprobs must be available!")

            # Build result with metadata
            raw_data = {
                'answer': response['answer'],
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'usage': response['usage'],
                'reasoning_content': response.get('reasoning_content', ''),
                'finish_reason': response.get('finish_reason', None),
                'temperature': TEMPERATURE,
                'max_tokens': MAX_TOKENS,
                'reasoning_effort': REASONING_EFFORT,
            }

            # Serialize logprobs with assertions
            if 'logprobs' in response and response['logprobs']:
                raw_data['logprobs'] = serialize_logprobs(
                    response['logprobs'],
                    predicted_answer,
                    response.get('reasoning_content', '')
                )

            return raw_data

        except Exception as e:
            error_msg = str(e).split('\n')[0][:200]
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


def main():
    logger.info(f"🚀 Starting benchmark: {MODEL_NAME} on {SPLIT} split")
    if LIMIT:
        logger.info(f"📊 Limit: {LIMIT} samples")

    # Load dataset
    dataset = get_med_qa_dataset()
    data = {'train': dataset.train, 'val': dataset.val, 'test': dataset.test}[SPLIT]

    if LIMIT:
        data = data.select(range(min(LIMIT, len(data))))

    # Setup output
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    model_safe = MODEL_NAME.replace('/', '_')
    json_file = output_path / f"{model_safe}_{SPLIT}.jsonl.gz"

    # Resume logic
    raw_results, completed_sample_ids = setup_resume_logic(json_file, len(data))

    questions_to_process = sum(1 for sample in data if get_sample_id_from_medqa(sample) not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"✅ All questions already completed!")
        stats = calculate_accuracy_summary(raw_results)
        log_accuracy_summary(stats)
        return raw_results

    logger.info(f"📝 Processing {questions_to_process}/{len(data)} questions with {NUM_THREADS} threads")

    SAVE_EVERY = 30

    def process_sample(idx, sample):
        import time
        sample_id = get_sample_id_from_medqa(sample)
        if sample_id in completed_sample_ids:
            return idx, None

        parsed = parse_medqa_sample(sample, idx)
        start_time = time.time()
        raw_data = run_inference(parsed['question'], parsed['options'], parsed['ground_truth'])
        raw_data['inference_time_seconds'] = time.time() - start_time

        # Add metadata
        raw_data['dataset_name'] = 'medqa'
        raw_data['split_name'] = SPLIT
        raw_data['question_index'] = idx
        raw_data['sample_id'] = parsed['sample_id']
        raw_data['question'] = parsed['question']
        raw_data['options'] = parsed['options']
        raw_data['meta_info'] = parsed.get('meta_info')

        return idx, raw_data

    # Run inference in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_sample, idx, sample): idx
                   for idx, sample in enumerate(data)
                   if get_sample_id_from_medqa(sample) not in completed_sample_ids}

        completed_count = 0
        with tqdm(total=len(data), desc=f"Running inference [{SPLIT}]", unit=" questions",
                  initial=len(data) - questions_to_process) as pbar:
            for future in as_completed(futures):
                try:
                    result_idx, raw_data = future.result()
                    if raw_data is not None:
                        raw_results[result_idx] = raw_data
                        completed_count += 1
                        pbar.update(1)

                        if completed_count % SAVE_EVERY == 0:
                            try:
                                completed_raw = [r for r in raw_results if r is not None]
                                atomic_save_json(completed_raw, json_file)
                                logger.info(f"💾 Checkpoint: {len(completed_raw)}/{len(data)}")
                            except Exception as save_err:
                                logger.error(f"⚠️  Checkpoint save failed: {save_err}")

                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Task {idx} failed: {str(e)}")
                    raw_results[idx] = {
                        'dataset_name': 'medqa',
                        'question_index': idx,
                        'answer': None,
                        'is_correct': False,
                        'error': str(e),
                    }
                    pbar.update(1)

    # Save final results
    logger.info("Saving final JSON")
    atomic_save_json(raw_results, json_file)

    # Calculate and log accuracy
    stats = calculate_accuracy_summary(raw_results)
    log_accuracy_summary(stats)
    logger.success(f"📁 Results saved to: {json_file}")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dataset: medqa/{SPLIT}")
    logger.info(f"Samples: {stats['total_count']}")
    logger.info(f"Accuracy: {stats['accuracy']:.2%}")

    return raw_results


if __name__ == "__main__":
    main()
