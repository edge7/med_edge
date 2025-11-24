"""
Clean benchmark script for medagents-benchmark using open-source models (vLLM).
Refactored to use shared utilities from benchmark_utils.py
"""

from pathlib import Path
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

from med_edge.llm_basic.vllm_native_request import (
    get_single_answer_vllm_native,
    extract_confidence_features,
    serialize_logprobs,
)
from med_edge.benchmark.benchmark_utils import (
    parse_medagents_sample,
    atomic_save_json,
    setup_resume_logic,
    calculate_accuracy_summary,
    log_accuracy_summary,
    get_sample_id_from_medagents,
)


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
ALL_CONFIGS = [
    "AfrimedQA", "MMLU", "MMLU-Pro", "MedBullets", "MedExQA",
    "MedMCQA", "MedXpertQA-R", "MedXpertQA-U", "PubMedQA"
]

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
BASE_URL = "http://192.222.53.56:8000/v1"
LIMIT = None  # Number of samples to test per config (None = all)
TEMPERATURE = 0.6  # 1.0 for OPENAI; 0.6 DeepSeek
MAX_TOKENS = 32768  # Max tokens for generation
REASONING_EFFORT = None  # None, "low", "mid", or "high" (for GPT-OSS only)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_test"
NUM_THREADS = 2  # Number of concurrent threads (2 is safe for vLLM)
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
                verbose=False,
            )

            predicted_answer = response['answer']
            is_correct = (predicted_answer == ground_truth)

            # Run feature extraction for validation/assertions only
            if 'logprobs' in response and response['logprobs']:
                valid_options = sorted(options.keys())
                _ = extract_confidence_features(response['logprobs'], predicted_answer, valid_options)
            else:
                raise Exception("Logprobs must be available!")

            # Build result
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


def run_single_config(dataset_config):
    """Run benchmark for a single config with resume support."""
    logger.info(f"🚀 Starting benchmark: {MODEL_NAME} on medagents-benchmark/{dataset_config}")
    if LIMIT:
        logger.info(f"📊 Limit: {LIMIT} samples")

    # Load dataset
    logger.info(f"📥 Loading medagents-benchmark config: {dataset_config}")
    test_data = load_dataset('super-dainiu/medagents-benchmark', dataset_config, split='test')
    test_hard_data = load_dataset('super-dainiu/medagents-benchmark', dataset_config, split='test_hard')

    hard_ids = set([q['realidx'] for q in test_hard_data])
    logger.info(f"📝 Loaded {len(test_data)} test questions ({len(hard_ids)} marked as hard)")

    if LIMIT:
        test_data = test_data.select(range(min(LIMIT, len(test_data))))

    # Setup output
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    model_safe = MODEL_NAME.replace('/', '_')
    json_file = output_path / f"{model_safe}_medagents_{dataset_config}_test.jsonl.gz"

    # Resume logic
    raw_results, completed_sample_ids = setup_resume_logic(json_file, len(test_data))

    questions_to_process = sum(1 for sample in test_data if sample['realidx'] not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"✅ All questions already completed for {dataset_config}!")
        return raw_results

    logger.info(f"📝 Processing {questions_to_process}/{len(test_data)} questions with {NUM_THREADS} threads")

    SAVE_EVERY = 30

    def process_sample(idx, sample):
        import time
        if sample['realidx'] in completed_sample_ids:
            return idx, None

        is_hard = sample['realidx'] in hard_ids
        parsed = parse_medagents_sample(sample, is_hard=is_hard)
        start_time = time.time()
        raw_data = run_inference(parsed['question'], parsed['options'], parsed['ground_truth'])
        raw_data['inference_time_seconds'] = time.time() - start_time

        # Add metadata
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

    # Run inference in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_sample, idx, sample): idx
                   for idx, sample in enumerate(test_data)
                   if sample['realidx'] not in completed_sample_ids}

        completed_count = 0
        with tqdm(total=len(test_data), desc=f"Running inference [{dataset_config}]", unit=" questions",
                  initial=len(test_data) - questions_to_process) as pbar:
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
                                logger.info(f"💾 Checkpoint: {len(completed_raw)}/{len(test_data)}")
                            except Exception as save_err:
                                logger.error(f"⚠️  Checkpoint save failed: {save_err}")

                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Task {idx} failed: {str(e)}")
                    raw_results[idx] = {
                        'dataset_name': 'medagents-benchmark',
                        'dataset_config': dataset_config,
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
    stats = calculate_accuracy_summary(raw_results, has_hard_subset=True)
    log_accuracy_summary(stats)
    logger.success(f"📁 Results saved to: {json_file}")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dataset: medagents-benchmark/{dataset_config}")
    logger.info(f"Samples: {stats['total_count']}")
    logger.info(f"Accuracy: {stats['accuracy']:.2%}")

    return raw_results


def main():
    """Main function to loop through all configs."""
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
            correct_count = sum(1 for r in results if r and r.get('is_correct', False))
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

    # Print final summary
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
