"""
Shared utilities for medical benchmark scripts.
Eliminates code duplication across benchmark runners.
"""

import json
import gzip
import hashlib
from pathlib import Path
from loguru import logger


# ============================================================================
# Dataset Parsing Functions
# ============================================================================

def parse_medqa_sample(sample, idx):
    """
    Parse a MedQA dataset sample.

    Args:
        sample: Dataset sample dict
        idx: Question index in the dataset

    Returns:
        dict with question, options, ground_truth, sample_id
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
    question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
    sample_id = int(question_hash[:16], 16)  # First 64 bits as int

    return {
        'question': question,
        'options': options,
        'ground_truth': ground_truth,
        'sample_id': sample_id,
        'question_index': idx,
        'meta_info': sample.get('meta_info', None),  # step1 or step2&3
    }


def parse_medagents_sample(sample, is_hard=False):
    """
    Parse a medagents-benchmark dataset sample.

    Args:
        sample: Dataset sample dict
        is_hard: Whether this question is marked as hard

    Returns:
        dict with question, options, ground_truth, sample_id, is_hard
    """
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


# ============================================================================
# File I/O Utilities
# ============================================================================

def atomic_save_json(data, file_path, compresslevel=6):
    """
    Atomically save data to a gzipped JSON file.
    Uses temp file + rename to prevent corruption.

    Args:
        data: Data to save (must be JSON-serializable)
        file_path: Path to save to (Path object or str)
        compresslevel: Gzip compression level (1-9)
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

    with gzip.open(temp_path, 'wt', encoding='utf-8', compresslevel=compresslevel) as f:
        json.dump(data, f)

    temp_path.replace(file_path)  # Atomic rename


def load_json_results(file_path):
    """
    Load results from a gzipped JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        tuple: (results_list, completed_sample_ids_set)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return [], set()

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            results = json.load(f)

        # Extract completed sample IDs
        completed_ids = set()
        for item in results:
            sample_id = item.get('sample_id')
            if sample_id is not None:
                completed_ids.add(sample_id)

        return results, completed_ids

    except Exception as e:
        logger.warning(f"⚠️  Could not load existing results: {e}")
        return [], set()


# ============================================================================
# Accuracy Calculation
# ============================================================================

def calculate_accuracy_summary(results, has_hard_subset=False):
    """
    Calculate accuracy from results list.

    Args:
        results: List of result dicts with 'is_correct' field
        has_hard_subset: Whether to calculate separate hard subset accuracy

    Returns:
        dict with accuracy metrics and formatted log messages
    """
    # Filter out None results
    completed_results = [r for r in results if r is not None]

    if not completed_results:
        return {
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'hard_accuracy': None,
            'hard_correct': 0,
            'hard_total': 0,
        }

    correct_count = sum(1 for r in completed_results if r.get('is_correct', False))
    total_count = len(completed_results)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    result = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'hard_accuracy': None,
        'hard_correct': 0,
        'hard_total': 0,
    }

    # Calculate hard subset accuracy if applicable
    if has_hard_subset:
        hard_results = [r for r in completed_results if r.get('is_hard', False)]
        if hard_results:
            hard_correct = sum(1 for r in hard_results if r.get('is_correct', False))
            hard_total = len(hard_results)
            result['hard_accuracy'] = hard_correct / hard_total
            result['hard_correct'] = hard_correct
            result['hard_total'] = hard_total

    return result


def log_accuracy_summary(accuracy_stats):
    """
    Log accuracy summary in a nice format.

    Args:
        accuracy_stats: Dict from calculate_accuracy_summary()
    """
    if accuracy_stats['hard_accuracy'] is not None:
        logger.success(
            f"\nAccuracy (all): {accuracy_stats['accuracy']:.2%} "
            f"({accuracy_stats['correct_count']}/{accuracy_stats['total_count']})"
        )
        logger.success(
            f"Accuracy (hard): {accuracy_stats['hard_accuracy']:.2%} "
            f"({accuracy_stats['hard_correct']}/{accuracy_stats['hard_total']})"
        )
    else:
        logger.success(
            f"\nAccuracy: {accuracy_stats['accuracy']:.2%} "
            f"({accuracy_stats['correct_count']}/{accuracy_stats['total_count']})"
        )


# ============================================================================
# Resume Logic Helpers
# ============================================================================

def setup_resume_logic(json_file, data_length):
    """
    Set up resume logic for a benchmark run.

    Args:
        json_file: Path to results JSON file
        data_length: Length of dataset (for pre-allocating results array)

    Returns:
        tuple: (raw_results array, completed_sample_ids set)
    """
    raw_results = [None] * data_length

    if json_file.exists():
        logger.info(f"📂 Found existing results file: {json_file}")
        existing_results, completed_ids = load_json_results(json_file)

        # Pre-populate results array
        for item in existing_results:
            idx = item.get('question_index')
            if idx is not None and 0 <= idx < data_length:
                raw_results[idx] = item

        if completed_ids:
            logger.info(f"✅ Found {len(completed_ids)} already completed questions - will skip these")

        return raw_results, completed_ids

    return raw_results, set()


def get_sample_id_from_medqa(sample):
    """Helper to extract sample_id from MedQA sample (using question hash)."""
    question = sample.get('question', '')
    question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
    return int(question_hash[:16], 16)


def get_sample_id_from_medagents(sample):
    """Helper to extract sample_id from medagents sample (using realidx)."""
    return sample.get('realidx', None)