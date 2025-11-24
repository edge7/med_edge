"""
Clean benchmark script for MedQA using proprietary models (OpenAI, Anthropic, etc.).
Refactored to use shared utilities from benchmark_utils.py
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

from med_edge.dataset_handler.hugging_face_handler import get_med_qa_dataset
from med_edge.llm_basic.generic_request import get_single_answer_benchmark
from med_edge.benchmark.benchmark_utils import parse_medqa_sample


# Load API keys from .env file
env_path = Path("/home/edge7/Desktop/projects/ing_edurso/luminai_backend/luminai_backend/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"✅ Loaded API keys from {env_path}")
else:
    logger.warning(f"⚠️  .env file not found at {env_path}")


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
MODEL = "google/gemini-3-pro-preview"  # e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022"
SPLIT = "test"  # train, val, or test
LIMIT = None  # Number of samples to test (None = all)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_proprietary"
VERBOSE = False  # Set to True to verify exact prompts being sent
# ============================================================================


def run_inference(question, options, ground_truth):
    """Run inference on a single question using instructor."""
    try:
        response = get_single_answer_benchmark(
            model=MODEL,
            question=question,
            options=options,
            verbose=VERBOSE,
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
        logger.error(f"❌ Error: {error_msg}")
        return {
            'predicted_answer': None,
            'ground_truth': ground_truth,
            'is_correct': False,
            'error': error_msg,
        }


def main():
    logger.info(f"🚀 Starting benchmark: {MODEL} on {SPLIT} split")
    if LIMIT:
        logger.info(f"📊 Limit: {LIMIT} samples")

    # Load dataset
    dataset = get_med_qa_dataset()
    data = {'train': dataset.train, 'val': dataset.val, 'test': dataset.test}[SPLIT]

    if LIMIT:
        data = data.select(range(min(LIMIT, len(data))))

    # Prepare output file (NO timestamp for resume support)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    model_safe = MODEL.replace('/', '_')
    csv_file = output_path / f"{model_safe}_{SPLIT}.csv"

    # Run inference on all samples
    results = []

    for idx, sample in enumerate(tqdm(data, desc=f"Running inference [{SPLIT}]", unit=" questions")):
        import time
        parsed = parse_medqa_sample(sample, idx)

        start_time = time.time()
        result = run_inference(
            question=parsed['question'],
            options=parsed['options'],
            ground_truth=parsed['ground_truth'],
        )
        result['inference_time_seconds'] = time.time() - start_time

        # Add metadata
        result['question_index'] = idx
        result['sample_id'] = parsed['sample_id']
        result['dataset_name'] = 'medqa'
        result['split_name'] = SPLIT
        result['model'] = MODEL
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

    logger.success(f"\n✅ Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    logger.success(f"📁 Results saved to: {csv_file}")

    return df


if __name__ == "__main__":
    main()
