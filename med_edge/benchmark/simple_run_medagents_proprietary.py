"""
Clean benchmark script for medagents-benchmark using proprietary models.
Refactored to use shared utilities from benchmark_utils.py
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv
from datasets import load_dataset

from med_edge.llm_basic.generic_request import get_single_answer_benchmark
from med_edge.benchmark.benchmark_utils import parse_medagents_sample


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
ALL_CONFIGS = [
    "AfrimedQA", "MMLU", "MMLU-Pro", "MedBullets", "MedExQA",
    "MedMCQA", "MedXpertQA-R", "MedXpertQA-U", "PubMedQA"
]

MODEL = "google/gemini-3-pro-preview"  # e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022"
LIMIT = None  # Number of samples to test per config (None = all)
OUTPUT_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_medagents_proprietary"
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


def run_single_config(dataset_config):
    """Run benchmark for a single config with resume support."""
    logger.info(f"🚀 Starting benchmark: {MODEL} on medagents-benchmark/{dataset_config}")
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

    model_safe = MODEL.replace('/', '_')
    base_filename = f"{model_safe}_medagents_{dataset_config}_test"
    csv_file = output_path / f"{base_filename}.csv"

    # ============================================================================
    # RESUME LOGIC: Load existing results and skip completed questions
    # ============================================================================
    completed_sample_ids = set()
    results = []

    if csv_file.exists():
        logger.info(f"📂 Found existing results file: {csv_file}")
        existing_df = pd.read_csv(csv_file)

        # Load completed sample IDs
        if 'sample_id' in existing_df.columns:
            completed_sample_ids = set(existing_df['sample_id'].dropna().tolist())
            logger.info(f"✅ Found {len(completed_sample_ids)} already completed questions - will skip these")

        # Load existing results
        results = existing_df.to_dict('records')

    # Count how many questions need processing
    questions_to_process = sum(1 for sample in test_data if sample['realidx'] not in completed_sample_ids)
    if questions_to_process == 0:
        logger.success(f"✅ All questions already completed for {dataset_config}!")
        df = pd.DataFrame(results)
        return df

    logger.info(f"📝 Processing {questions_to_process}/{len(test_data)} questions ({len(test_data) - questions_to_process} already done)")

    SAVE_EVERY = 30  # Save checkpoint every 30 samples

    # Run inference on all samples
    completed_count = 0
    with tqdm(total=len(test_data),
              desc=f"Running inference [{dataset_config}]",
              unit=" questions",
              initial=len(test_data) - questions_to_process) as pbar:
        for idx, sample in enumerate(test_data):
            # Skip if already completed
            if sample['realidx'] in completed_sample_ids:
                continue

            # Check if this is a hard question
            is_hard = sample['realidx'] in hard_ids

            parsed = parse_medagents_sample(sample, is_hard=is_hard)

            import time
            start_time = time.time()
            result = run_inference(
                question=parsed['question'],
                options=parsed['options'],
                ground_truth=parsed['ground_truth'],
            )
            result['inference_time_seconds'] = time.time() - start_time

            # Add metadata
            result['dataset_name'] = 'medagents-benchmark'
            result['dataset_config'] = dataset_config
            result['split_name'] = 'test'
            result['question_index'] = idx
            result['sample_id'] = parsed['sample_id']
            result['is_hard'] = parsed['is_hard']
            result['model'] = MODEL
            result['question'] = parsed['question']
            result['meta_info'] = parsed['meta_info']

            # Add all option columns (handle variable number of options)
            for opt_key in ['a', 'b', 'c', 'd', 'e']:
                result[f'option_{opt_key}'] = parsed['options'].get(opt_key, '')

            results.append(result)
            completed_count += 1
            pbar.update(1)

            # Save checkpoint every SAVE_EVERY samples
            if completed_count % SAVE_EVERY == 0:
                try:
                    # ATOMIC WRITE: Save to temp file first, then rename
                    csv_temp = csv_file.with_suffix('.csv.tmp')
                    df_temp = pd.DataFrame(results)
                    df_temp.to_csv(csv_temp, index=False)
                    csv_temp.replace(csv_file)  # Atomic rename
                    logger.info(f"💾 Checkpoint saved: {len(results)}/{len(test_data)} samples")
                except Exception as save_err:
                    logger.error(f"⚠️  Checkpoint save failed (continuing): {save_err}")

    # Save final CSV atomically
    csv_temp = csv_file.with_suffix('.csv.tmp')
    df = pd.DataFrame(results)
    df.to_csv(csv_temp, index=False)
    csv_temp.replace(csv_file)  # Atomic rename

    # Calculate accuracy
    accuracy = df['is_correct'].mean()

    # Calculate hard subset accuracy
    if 'is_hard' in df.columns:
        hard_df = df[df['is_hard'] == True]
        if len(hard_df) > 0:
            hard_accuracy = hard_df['is_correct'].mean()
            logger.success(f"\nAccuracy (all): {accuracy:.2%} ({df['is_correct'].sum()}/{len(df)})")
            logger.success(f"Accuracy (hard): {hard_accuracy:.2%} ({hard_df['is_correct'].sum()}/{len(hard_df)})")
        else:
            logger.success(f"\nAccuracy: {accuracy:.2%} ({df['is_correct'].sum()}/{len(df)})")
    else:
        logger.success(f"\nAccuracy: {accuracy:.2%} ({df['is_correct'].sum()}/{len(df)})")

    logger.success(f"📁 Results saved to: {csv_file}")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Dataset: medagents-benchmark/{dataset_config}")
    logger.info(f"Samples: {len(df)}")
    logger.info(f"Accuracy: {accuracy:.2%}")

    return df


def main():
    """Main function to loop through all configs with resume support."""
    logger.info("="*80)
    logger.info(f"🔬 Running ALL medagents-benchmark configs with proprietary model")
    logger.info(f"   Model: {MODEL}")
    logger.info(f"   Total configs: {len(ALL_CONFIGS)}")
    logger.info(f"   Configs: {', '.join(ALL_CONFIGS)}")
    logger.info("="*80)

    results_summary = {}

    for idx, config in enumerate(ALL_CONFIGS, 1):
        logger.info(f"\n\n{'='*80}")
        logger.info(f"📌 CONFIG {idx}/{len(ALL_CONFIGS)}: {config}")
        logger.info(f"{'='*80}\n")

        try:
            df = run_single_config(config)
            accuracy = df['is_correct'].mean()
            results_summary[config] = {
                'samples': len(df),
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
