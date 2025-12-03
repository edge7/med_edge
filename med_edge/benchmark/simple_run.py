"""
Clean benchmark script for MedQA using open-source models (vLLM).
Refactored to use shared utilities from benchmark_utils.py

Usage:
    python -m med_edge.benchmark.simple_run --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --base-url "http://localhost:8000/v1"
    python -m med_edge.benchmark.simple_run --help
"""

import click
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
    append_jsonl,
    setup_resume_logic_jsonl,
    get_sample_id_from_medqa,
)


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
                ask_prob=config.get('ask_prob', False),
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
                'temperature': config['temperature'],
                'max_tokens': config['max_tokens'],
                'reasoning_effort': config['reasoning_effort'],
                'model_name': config['model_name'],
            }

            # Add verbalized confidence if available
            if 'verbalized_confidence' in response:
                raw_data['verbalized_confidence'] = response['verbalized_confidence']

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


@click.command()
@click.option('--model', '-m', required=True, help='Model name (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")')
@click.option('--base-url', '-u', required=True, help='vLLM server URL (e.g., "http://localhost:8000/v1")')
@click.option('--split', '-s', type=click.Choice(['train', 'val', 'test']), default='test', help='Dataset split')
@click.option('--limit', '-l', type=int, default=None, help='Limit number of samples (default: all)')
@click.option('--temperature', '-t', type=float, default=0.6, help='Sampling temperature (default: 0.6)')
@click.option('--max-tokens', type=int, default=32768, help='Max tokens for generation (default: 32768)')
@click.option('--reasoning-effort', type=click.Choice(['low', 'medium', 'high']), default=None, help='Reasoning effort level')
@click.option('--output-dir', '-o', type=click.Path(), required=True, help='Output directory for results')
@click.option('--threads', type=int, default=4, help='Number of concurrent threads (default: 3)')
@click.option('--verbose', '-v', is_flag=True, help='Show exact prompts being sent')
@click.option('--ask-prob', is_flag=True, default=False, help='Ask model to provide verbalized confidence score (1-100) with the answer')
def main(model, base_url, split, limit, temperature, max_tokens, reasoning_effort, output_dir, threads, verbose, ask_prob):
    """Run MedQA benchmark with open-source models via vLLM."""

    # Build config dict for passing to functions
    config = {
        'model_name': model,
        'base_url': base_url,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'reasoning_effort': reasoning_effort,
        'verbose': verbose,
        'ask_prob': ask_prob,
    }

    logger.info(f"Starting benchmark: {model} on {split} split")
    if limit:
        logger.info(f"Limit: {limit} samples")

    # Load dataset
    dataset = get_med_qa_dataset()
    data = {'train': dataset.train, 'val': dataset.val, 'test': dataset.test}[split]

    if limit:
        data = data.select(range(min(limit, len(data))))

    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_safe = model.replace('/', '_')
    jsonl_file = output_path / f"{model_safe}_{split}_raw.jsonl"

    # Resume logic - memory efficient, only loads sample IDs
    completed_sample_ids = setup_resume_logic_jsonl(jsonl_file)

    questions_to_process = sum(1 for sample in data if get_sample_id_from_medqa(sample) not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"All questions already completed!")
        return

    logger.info(f"Processing {questions_to_process}/{len(data)} questions with {threads} threads")

    def process_sample(idx, sample):
        import time
        sample_id = get_sample_id_from_medqa(sample)
        if sample_id in completed_sample_ids:
            return idx, None

        parsed = parse_medqa_sample(sample, idx)
        start_time = time.time()
        raw_data = run_inference(parsed['question'], parsed['options'], parsed['ground_truth'], config)
        raw_data['inference_time_seconds'] = time.time() - start_time

        # Add metadata
        raw_data['dataset_name'] = 'medqa'
        raw_data['split_name'] = split
        raw_data['question_index'] = idx
        raw_data['sample_id'] = parsed['sample_id']
        raw_data['question'] = parsed['question']
        raw_data['options'] = parsed['options']
        raw_data['meta_info'] = parsed.get('meta_info')

        return idx, raw_data

    # Run inference in parallel
    completed_count = len(completed_sample_ids)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_sample, idx, sample): idx
                   for idx, sample in enumerate(data)
                   if get_sample_id_from_medqa(sample) not in completed_sample_ids}

        with tqdm(total=len(data), desc=f"Running inference [{split}]", unit=" questions",
                  initial=len(data) - questions_to_process) as pbar:
            for future in as_completed(futures):
                try:
                    result_idx, raw_data = future.result()
                    if raw_data is not None:
                        # Append immediately to JSONL - memory efficient
                        append_jsonl(raw_data, jsonl_file)
                        completed_count += 1
                        pbar.update(1)

                        if completed_count % 30 == 0:
                            logger.info(f"Progress: {completed_count}/{len(data)}")

                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Task {idx} failed: {str(e)}")
                    error_result = {
                        'dataset_name': 'medqa',
                        'question_index': idx,
                        'sample_id': get_sample_id_from_medqa(data[idx]),
                        'answer': None,
                        'is_correct': False,
                        'error': str(e),
                    }
                    append_jsonl(error_result, jsonl_file)
                    pbar.update(1)

    logger.success(f"Results saved to: {jsonl_file}")


if __name__ == "__main__":
    main()
