"""
Simple benchmark script for proprietary models (OpenAI, Anthropic, etc.)
Uses instructor for structured outputs - much simpler than the vLLM version.
Edit the config section below and execute.
"""

import os
import pandas as pd
import hashlib
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

# Load API keys from .env file
env_path = Path("/home/edge7/Desktop/projects/ing_edurso/luminai_backend/luminai_backend/.env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"✅ Loaded API keys from {env_path}")
else:
    logger.warning(f"⚠️  .env file not found at {env_path}")

from med_edge.dataset_handler.hugging_face_handler import get_med_qa_dataset
from med_edge.llm_basic.generic_request import get_single_answer_benchmark


# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
MODEL = "google/gemini-3-pro-preview"  # e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022"
SPLIT = "test"  # train, val, or test
LIMIT = None  # Number of samples to test (None = all)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_proprietary"
VERBOSE = False  # Set to True to verify exact prompts being sent (for research verification)
# ============================================================================


def parse_medqa_sample(sample, idx):
    """
    Parse a MedQA dataset sample.

    Args:
        sample: Dataset sample dict
        idx: Question index in the dataset
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
        'question_index': idx,
    }


def run_inference(question, options, ground_truth):
    """
    Run inference on a single question using instructor.

    Returns:
        dict: Result with answer and correctness
    """
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
        error_msg = str(e).split('\n')[0][:200]  # First line, max 200 chars
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

    if SPLIT == 'train':
        data = dataset.train
    elif SPLIT == 'val':
        data = dataset.val
    else:
        data = dataset.test

    # Limit if specified
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
        parsed = parse_medqa_sample(sample, idx)

        result = run_inference(
            question=parsed['question'],
            options=parsed['options'],
            ground_truth=parsed['ground_truth'],
        )

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