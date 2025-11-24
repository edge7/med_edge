#!/usr/bin/env python
"""
CRITICAL VERIFICATION SCRIPT FOR RESEARCH DATA INTEGRITY
Run this on any benchmark results file to ensure all logprobs data is preserved.

Usage:
    python validate_logprobs.py results_file.jsonl.gz

This script will:
1. Load each result from the file
2. Verify ALL tokens have exactly 20 top_logprobs
3. Verify answer probabilities are extracted correctly
4. Report any missing or corrupted data
5. RAISE EXCEPTIONS if any data is missing
"""

import gzip
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.logprobs_validator import (
    validate_serialized_logprobs,
    validate_benchmark_result,
    extract_answer_probabilities
)


def check_single_result(result: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Thoroughly check a single benchmark result.

    Args:
        result: The result dictionary to check
        idx: Index in the file for error reporting

    Returns:
        Summary statistics about the result

    Raises:
        AssertionError: If ANY data is missing or corrupted
    """
    print(f"\n🔍 Checking result {idx + 1}...")

    # Basic structure validation
    required_fields = ['question', 'model_answer', 'serialized_logprobs', 'correct_answer']
    for field in required_fields:
        assert field in result, f"Result {idx} missing required field: {field}"

    # Validate logprobs data
    logprobs = result['serialized_logprobs']
    validation_result = validate_serialized_logprobs(logprobs)

    if not validation_result['valid']:
        raise AssertionError(
            f"Result {idx} has INVALID logprobs data:\n"
            f"  Errors: {validation_result['errors']}\n"
            f"  This means data loss occurred during serialization!"
        )

    # Extract statistics
    num_tokens = validation_result['num_tokens']
    num_with_top_logprobs = validation_result['num_tokens_with_top_logprobs']

    print(f"  ✓ Total tokens: {num_tokens}")
    print(f"  ✓ Tokens with full top_logprobs: {num_with_top_logprobs}")

    # Check that ALL tokens have top_logprobs (for answer tokens)
    answer_token_indices = []
    for i, token_data in enumerate(logprobs):
        token = token_data['token']
        if token.lower() in ['a', 'b', 'c', 'd', 'e']:
            answer_token_indices.append(i)
            # This token MUST have top_logprobs
            if 'top_logprobs' not in token_data or not token_data['top_logprobs']:
                raise AssertionError(
                    f"Result {idx}: Answer token '{token}' at position {i} "
                    f"is MISSING top_logprobs! This is CRITICAL data loss!"
                )

            num_top = len(token_data['top_logprobs'])
            if num_top != 20:
                raise AssertionError(
                    f"Result {idx}: Answer token '{token}' has {num_top} top_logprobs, "
                    f"expected 20! Data loss detected!"
                )

            print(f"  ✓ Answer token '{token}' at position {i} has all 20 top_logprobs")

    # Extract answer probabilities
    try:
        answer_probs = extract_answer_probabilities(logprobs, result['model_answer'])

        if not answer_probs:
            print(f"  ⚠️ WARNING: Could not extract answer probabilities")
        else:
            print(f"  ✓ Answer probabilities extracted successfully:")
            for option, prob_data in answer_probs.items():
                if prob_data:
                    prob = math.exp(prob_data['logprob'])
                    print(f"    Option {option}: {prob:.4%} (logprob: {prob_data['logprob']:.4f})")
    except Exception as e:
        raise AssertionError(
            f"Result {idx}: Failed to extract answer probabilities: {str(e)}\n"
            f"This indicates corrupted or missing logprobs data!"
        )

    # Full result validation
    full_validation = validate_benchmark_result(result)
    if not full_validation['valid']:
        raise AssertionError(
            f"Result {idx} FAILED full validation:\n"
            f"  Errors: {full_validation['errors']}"
        )

    print(f"  ✅ Result {idx + 1} is VALID with all data preserved!")

    return {
        'idx': idx,
        'num_tokens': num_tokens,
        'num_answer_tokens': len(answer_token_indices),
        'has_answer_probs': bool(answer_probs),
        'model_answer': result['model_answer'],
        'correct': result['model_answer'] == result['correct_answer']
    }


def check_results_file(filepath: str):
    """
    Check all results in a benchmark output file.

    Args:
        filepath: Path to the results file (jsonl or jsonl.gz)

    Raises:
        AssertionError: If ANY result has missing or corrupted data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"=" * 80)
    print(f"🔬 RESEARCH DATA INTEGRITY CHECK")
    print(f"=" * 80)
    print(f"File: {filepath}")
    print(f"Size: {filepath.stat().st_size / 1024:.1f} KB")
    print(f"=" * 80)

    # Determine how to open the file
    if filepath.suffix == '.gz':
        open_func = lambda: gzip.open(filepath, 'rt')
    else:
        open_func = lambda: open(filepath, 'r')

    results = []
    stats = []
    errors = []

    with open_func() as f:
        for idx, line in enumerate(f):
            try:
                result = json.loads(line.strip())
                result_stats = check_single_result(result, idx)
                stats.append(result_stats)
                results.append(result)
            except AssertionError as e:
                error_msg = str(e)
                print(f"\n❌ ERROR at result {idx + 1}: {error_msg}")
                errors.append({'idx': idx, 'error': error_msg})
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                print(f"\n❌ ERROR at line {idx + 1}: {error_msg}")
                errors.append({'idx': idx, 'error': error_msg})
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"\n❌ ERROR at result {idx + 1}: {error_msg}")
                errors.append({'idx': idx, 'error': error_msg})

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total results checked: {len(stats) + len(errors)}")
    print(f"✅ Valid results: {len(stats)}")
    print(f"❌ Invalid results: {len(errors)}")

    if stats:
        total_tokens = sum(s['num_tokens'] for s in stats)
        total_answer_tokens = sum(s['num_answer_tokens'] for s in stats)
        correct = sum(1 for s in stats if s['correct'])

        print(f"\n📈 Valid Results Statistics:")
        print(f"  - Total tokens processed: {total_tokens:,}")
        print(f"  - Total answer tokens: {total_answer_tokens:,}")
        print(f"  - Average tokens per result: {total_tokens/len(stats):.1f}")
        print(f"  - Results with answer probabilities: {sum(1 for s in stats if s['has_answer_probs'])}/{len(stats)}")
        print(f"  - Accuracy: {correct}/{len(stats)} ({100*correct/len(stats):.1f}%)")

    if errors:
        print(f"\n⚠️ CRITICAL ERRORS FOUND:")
        print(f"The following results have MISSING or CORRUPTED logprobs data:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - Result {error['idx'] + 1}: {error['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

        raise AssertionError(
            f"\n🚨 DATA INTEGRITY CHECK FAILED!\n"
            f"{len(errors)} results have missing or corrupted logprobs data.\n"
            f"This is CRITICAL for research - the benchmark results are NOT valid!"
        )
    else:
        print(f"\n✅ ALL CHECKS PASSED!")
        print(f"All {len(stats)} results have complete logprobs data preserved.")
        print(f"The data is suitable for research use.")

    return results, stats


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python validate_logprobs.py <results_file.jsonl.gz>")
        print("\nThis script validates that ALL logprobs data is preserved in benchmark results.")
        print("It will RAISE EXCEPTIONS if any data is missing - critical for research!")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        results, stats = check_results_file(filepath)
        print(f"\n🎉 Success! File '{filepath}' is valid for research use.")
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()