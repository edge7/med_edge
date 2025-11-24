"""
Clean benchmark script for medagents-benchmark using open-source models (vLLM).
Refactored to use shared utilities from benchmark_utils.py

Usage:
    python -m med_edge.benchmark.simple_run_medagents --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --base-url "http://localhost:8000/v1" --output-dir ./results
    python -m med_edge.benchmark.simple_run_medagents --model "..." --base-url "..." --output-dir ./results --configs MMLU MedMCQA
    python -m med_edge.benchmark.simple_run_medagents --help
"""

import click
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
)


ALL_CONFIGS = [
    "AfrimedQA", "MMLU", "MMLU-Pro", "MedBullets", "MedExQA",
    "MedMCQA", "MedXpertQA-R", "MedXpertQA-U", "PubMedQA"
]


def run_inference(question, options, ground_truth, config, max_retries=2):
    """Run inference with retry logic."""
    import time

    for attempt in range(max_retries + 1):
        try:
            response = get_single_answer_vllm_native(
                model_name=config['model_name'],
                question=question,
                options=options,
                base_url=config['base_url'],
                api_key="EMPTY",
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                reasoning_effort=config['reasoning_effort'],
                verbose=config['verbose'],
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
                'temperature': config['temperature'],
                'max_tokens': config['max_tokens'],
                'reasoning_effort': config['reasoning_effort'],
                'model_name': config['model_name'],
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
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}. Retrying in 2s...")
                time.sleep(2)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {error_msg}")
                return {
                    'answer': None,
                    'ground_truth': ground_truth,
                    'is_correct': False,
                    'error': error_msg,
                }
    return None


def run_single_config(dataset_config, config, output_dir, threads, limit):
    """Run benchmark for a single config with resume support."""
    logger.info(f"Starting benchmark: {config['model_name']} on medagents-benchmark/{dataset_config}")
    if limit:
        logger.info(f"Limit: {limit} samples")

    # Load dataset
    logger.info(f"Loading medagents-benchmark config: {dataset_config}")
    test_data = load_dataset('super-dainiu/medagents-benchmark', dataset_config, split='test')
    test_hard_data = load_dataset('super-dainiu/medagents-benchmark', dataset_config, split='test_hard')

    hard_ids = set([q['realidx'] for q in test_hard_data])
    logger.info(f"Loaded {len(test_data)} test questions ({len(hard_ids)} marked as hard)")

    if limit:
        test_data = test_data.select(range(min(limit, len(test_data))))

    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_safe = config['model_name'].replace('/', '_')
    json_file = output_path / f"{model_safe}_medagents_{dataset_config}_test_raw.jsonl.gz"

    # Resume logic
    raw_results, completed_sample_ids = setup_resume_logic(json_file, len(test_data))

    questions_to_process = sum(1 for sample in test_data if sample['realidx'] not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"All questions already completed for {dataset_config}!")
        return raw_results

    logger.info(f"Processing {questions_to_process}/{len(test_data)} questions with {threads} threads")

    SAVE_EVERY = 30

    def process_sample(idx, sample):
        import time
        if sample['realidx'] in completed_sample_ids:
            return idx, None

        is_hard = sample['realidx'] in hard_ids
        parsed = parse_medagents_sample(sample, is_hard=is_hard)
        start_time = time.time()
        raw_data = run_inference(parsed['question'], parsed['options'], parsed['ground_truth'], config)
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
    with ThreadPoolExecutor(max_workers=threads) as executor:
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
                                logger.info(f"Checkpoint: {len(completed_raw)}/{len(test_data)}")
                            except Exception as save_err:
                                logger.error(f"Checkpoint save failed: {save_err}")

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
    logger.success(f"Results saved to: {json_file}")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dataset: medagents-benchmark/{dataset_config}")
    logger.info(f"Samples: {stats['total_count']}")
    logger.info(f"Accuracy: {stats['accuracy']:.2%}")

    return raw_results


@click.command()
@click.option('--model', '-m', required=True, help='Model name (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")')
@click.option('--base-url', '-u', required=True, help='vLLM server URL (e.g., "http://localhost:8000/v1")')
@click.option('--configs', '-c', multiple=True, default=None,
              help=f'Dataset configs to run. Can specify multiple. Default: all. Available: {", ".join(ALL_CONFIGS)}')
@click.option('--limit', '-l', type=int, default=None, help='Limit number of samples per config (default: all)')
@click.option('--temperature', '-t', type=float, default=0.6, help='Sampling temperature (default: 0.6)')
@click.option('--max-tokens', type=int, default=22768, help='Max tokens for generation (default: 32768)')
@click.option('--reasoning-effort', type=click.Choice(['low', 'mid', 'high']), default=None, help='Reasoning effort level')
@click.option('--output-dir', '-o', type=click.Path(), required=True, help='Output directory for results')
@click.option('--threads', type=int, default=2, help='Number of concurrent threads (default: 2)')
@click.option('--verbose', '-v', is_flag=True, help='Show exact prompts being sent')
def main(model, base_url, configs, limit, temperature, max_tokens, reasoning_effort, output_dir, threads, verbose):
    """Run medagents-benchmark with open-source models via vLLM."""

    # Build config dict for passing to functions
    config = {
        'model_name': model,
        'base_url': base_url,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'reasoning_effort': reasoning_effort,
        'verbose': verbose,
    }

    # Determine which configs to run
    configs_to_run = list(configs) if configs else ALL_CONFIGS

    # Validate configs
    invalid_configs = set(configs_to_run) - set(ALL_CONFIGS)
    if invalid_configs:
        raise click.BadParameter(f"Invalid configs: {invalid_configs}. Available: {ALL_CONFIGS}")

    logger.info("=" * 80)
    logger.info(f"Running medagents-benchmark configs")
    logger.info(f"   Model: {model}")
    logger.info(f"   Total configs: {len(configs_to_run)}")
    logger.info(f"   Configs: {', '.join(configs_to_run)}")
    logger.info("=" * 80)

    results_summary = {}

    for idx, dataset_config in enumerate(configs_to_run, 1):
        logger.info(f"\n\n{'=' * 80}")
        logger.info(f"CONFIG {idx}/{len(configs_to_run)}: {dataset_config}")
        logger.info(f"{'=' * 80}\n")

        try:
            results = run_single_config(dataset_config, config, output_dir, threads, limit)
            correct_count = sum(1 for r in results if r and r.get('is_correct', False))
            accuracy = correct_count / len(results) if len(results) > 0 else 0.0
            results_summary[dataset_config] = {
                'samples': len(results),
                'accuracy': accuracy,
                'status': 'completed'
            }
        except Exception as e:
            logger.error(f"Failed to process config {dataset_config}: {str(e)}")
            results_summary[dataset_config] = {
                'samples': 0,
                'accuracy': 0.0,
                'status': 'failed',
                'error': str(e)
            }

    # Print final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("FINAL SUMMARY - ALL CONFIGS")
    logger.info("=" * 80)
    for dataset_config, stats in results_summary.items():
        if stats['status'] == 'completed':
            logger.info(f"  {dataset_config:15s}: {stats['accuracy']:.2%} ({stats['samples']} samples)")
        else:
            logger.error(f"  {dataset_config:15s}: FAILED - {stats.get('error', 'Unknown error')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
