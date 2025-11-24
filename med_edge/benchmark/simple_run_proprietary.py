"""
Clean benchmark script for MedQA using proprietary models (OpenAI, Anthropic, etc.).
Refactored to use shared utilities from benchmark_utils.py

Usage:
    python -m med_edge.benchmark.simple_run_proprietary --model "openai/gpt-4" --output-dir ./results
    python -m med_edge.benchmark.simple_run_proprietary --model "anthropic/claude-3-5-sonnet-20241022" --output-dir ./results --env-file .env
    python -m med_edge.benchmark.simple_run_proprietary --help
"""

import click
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

from med_edge.dataset_handler.hugging_face_handler import get_med_qa_dataset
from med_edge.llm_basic.generic_request import get_single_answer_benchmark
from med_edge.benchmark.benchmark_utils import parse_medqa_sample


def run_inference(question, options, ground_truth, config):
    """Run inference on a single question using instructor."""
    try:
        response = get_single_answer_benchmark(
            model=config['model'],
            question=question,
            options=options,
            verbose=config['verbose'],
        )

        predicted_answer = response['answer']
        is_correct = (predicted_answer == ground_truth)

        return {
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'error': None,
        }

    except Exception as e:
        error_msg = str(e).split('\n')[0][:200]
        logger.error(f"Error: {error_msg}")
        return {
            'predicted_answer': None,
            'ground_truth': ground_truth,
            'is_correct': False,
            'error': error_msg,
        }


@click.command()
@click.option('--model', '-m', required=True, help='Model name (e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022")')
@click.option('--split', '-s', type=click.Choice(['train', 'val', 'test']), default='test', help='Dataset split')
@click.option('--limit', '-l', type=int, default=None, help='Limit number of samples (default: all)')
@click.option('--output-dir', '-o', type=click.Path(), required=True, help='Output directory for results')
@click.option('--env-file', type=click.Path(exists=True), default=None, help='Path to .env file with API keys')
@click.option('--verbose', '-v', is_flag=True, help='Show exact prompts being sent')
def main(model, split, limit, output_dir, env_file, verbose):
    """Run MedQA benchmark with proprietary models (OpenAI, Anthropic, etc.)."""

    # Load environment variables if env file provided
    if env_file:
        load_dotenv(dotenv_path=env_file)
        logger.info(f"Loaded API keys from {env_file}")

    # Build config dict
    config = {
        'model': model,
        'verbose': verbose,
    }

    logger.info(f"Starting benchmark: {model} on {split} split")
    if limit:
        logger.info(f"Limit: {limit} samples")

    # Load dataset
    dataset = get_med_qa_dataset()
    data = {'train': dataset.train, 'val': dataset.val, 'test': dataset.test}[split]

    if limit:
        data = data.select(range(min(limit, len(data))))

    # Prepare output file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_safe = model.replace('/', '_')
    csv_file = output_path / f"{model_safe}_{split}.csv"

    # Run inference on all samples
    results = []

    for idx, sample in enumerate(tqdm(data, desc=f"Running inference [{split}]", unit=" questions")):
        import time
        parsed = parse_medqa_sample(sample, idx)

        start_time = time.time()
        result = run_inference(
            question=parsed['question'],
            options=parsed['options'],
            ground_truth=parsed['ground_truth'],
            config=config,
        )
        result['inference_time_seconds'] = time.time() - start_time

        # Add metadata
        result['question_index'] = idx
        result['sample_id'] = parsed['sample_id']
        result['dataset_name'] = 'medqa'
        result['split_name'] = split
        result['model'] = model
        result['question'] = parsed['question']

        # Add options as separate columns
        for opt_key in ['a', 'b', 'c', 'd', 'e']:
            result[f'option_{opt_key}'] = parsed['options'].get(opt_key, '')

        results.append(result)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)

    # Calculate accuracy
    accuracy = df['is_correct'].mean()
    correct_count = df['is_correct'].sum()
    total_count = len(df)

    logger.success(f"\nAccuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    logger.success(f"Results saved to: {csv_file}")

    return df


if __name__ == "__main__":
    main()
