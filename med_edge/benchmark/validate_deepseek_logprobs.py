#!/usr/bin/env python
"""
Validation script for DeepSeek benchmark results with logprobs.
Specifically for the format used in the benchmarks_test directory.

Usage:
    python validate_deepseek_logprobs.py results_file.jsonl.gz
"""

import gzip
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List


def validate_token_logprobs(token_data: Dict, token_idx: int) -> List[str]:
    """
    Validate a single token's logprobs data.

    Returns list of errors (empty if valid).
    """
    errors = []

    # Check required fields
    if 'token' not in token_data:
        errors.append(f"Token {token_idx}: missing 'token' field")

    if 'logprob' not in token_data:
        errors.append(f"Token {token_idx}: missing 'logprob' field")

    # Check top_logprobs
    if 'top_logprobs' not in token_data:
        errors.append(f"Token {token_idx}: missing 'top_logprobs' field")
    else:
        top_logprobs = token_data['top_logprobs']
        if not isinstance(top_logprobs, list):
            errors.append(f"Token {token_idx}: top_logprobs is not a list")
        else:
            # Check if we have sufficient top_logprobs
            num_top = len(top_logprobs)
            if num_top < 5:  # Minimum expectation for answer tokens
                errors.append(f"Token {token_idx}: only {num_top} top_logprobs (expected at least 5)")

    return errors


def extract_answer_probabilities_deepseek(logprobs_content: List[Dict], model_answer: str) -> Dict[str, Dict]:
    """
    Extract answer probabilities from DeepSeek format logprobs.
    """
    answer_probs = {}
    answer_options = ['a', 'b', 'c', 'd', 'e']

    # Look for answer tokens
    for token_data in logprobs_content:
        token = token_data.get('token', '').lower().strip()

        # Check if this is an answer token
        if token in answer_options:
            # Store the token's own logprob
            if token not in answer_probs:
                answer_probs[token] = {
                    'logprob': token_data.get('logprob'),
                    'is_selected': token == model_answer.lower()
                }

            # Also check top_logprobs for other answer options
            if 'top_logprobs' in token_data:
                for alt in token_data['top_logprobs']:
                    alt_token = alt.get('token', '').lower().strip()
                    if alt_token in answer_options and alt_token not in answer_probs:
                        answer_probs[alt_token] = {
                            'logprob': alt.get('logprob'),
                            'is_selected': False
                        }

    return answer_probs


def check_single_result(result: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Check a single DeepSeek benchmark result.
    """
    print(f"\n🔍 Checking result {idx + 1}...")

    # Check required fields for DeepSeek format
    required_fields = ['answer', 'ground_truth', 'is_correct', 'logprobs']
    missing_fields = []

    for field in required_fields:
        if field not in result:
            missing_fields.append(field)

    if missing_fields:
        raise AssertionError(f"Result {idx}: Missing required fields: {missing_fields}")

    # Validate logprobs structure
    logprobs = result['logprobs']
    if 'content' not in logprobs:
        raise AssertionError(f"Result {idx}: logprobs missing 'content' field")

    content = logprobs['content']
    if not isinstance(content, list):
        raise AssertionError(f"Result {idx}: logprobs content is not a list")

    num_tokens = len(content)
    print(f"  ✓ Total tokens: {num_tokens}")

    # Check each token
    all_errors = []
    num_with_top_logprobs = 0
    answer_tokens_found = []

    for i, token_data in enumerate(content):
        errors = validate_token_logprobs(token_data, i)
        if errors:
            all_errors.extend(errors)

        # Count tokens with top_logprobs
        if 'top_logprobs' in token_data and len(token_data.get('top_logprobs', [])) > 0:
            num_with_top_logprobs += 1

        # Check if this is an answer token
        token = token_data.get('token', '').lower().strip()
        if token in ['a', 'b', 'c', 'd', 'e']:
            answer_tokens_found.append(token)

            # For answer tokens, ensure we have top_logprobs
            if 'top_logprobs' not in token_data or not token_data['top_logprobs']:
                raise AssertionError(
                    f"Result {idx}: Answer token '{token}' at position {i} "
                    f"is MISSING top_logprobs! Critical for research!"
                )

            num_top = len(token_data['top_logprobs'])
            print(f"  ✓ Answer token '{token}' at position {i} has {num_top} top_logprobs")

    print(f"  ✓ Tokens with top_logprobs: {num_with_top_logprobs}/{num_tokens}")

    if answer_tokens_found:
        print(f"  ✓ Answer tokens found: {answer_tokens_found}")
    else:
        print(f"  ⚠️ No explicit answer tokens found in logprobs")

    # Extract answer probabilities
    try:
        answer_probs = extract_answer_probabilities_deepseek(content, result['answer'])

        if answer_probs:
            print(f"  ✓ Answer probabilities extracted:")
            for option, prob_data in sorted(answer_probs.items()):
                if prob_data and prob_data['logprob'] is not None:
                    prob = math.exp(prob_data['logprob'])
                    selected = " ← SELECTED" if prob_data.get('is_selected') else ""
                    print(f"    Option {option.upper()}: {prob:.4%} (logprob: {prob_data['logprob']:.4f}){selected}")
        else:
            print(f"  ⚠️ Could not extract answer probabilities")
    except Exception as e:
        print(f"  ⚠️ Error extracting answer probabilities: {e}")

    # Check if any critical errors
    if all_errors:
        print(f"  ⚠️ Validation issues found: {len(all_errors)} warnings")
        for error in all_errors[:3]:
            print(f"    - {error}")

    print(f"  ✅ Result {idx + 1} validated successfully!")

    return {
        'idx': idx,
        'num_tokens': num_tokens,
        'num_with_top_logprobs': num_with_top_logprobs,
        'answer_tokens_found': len(answer_tokens_found),
        'model_answer': result['answer'],
        'ground_truth': result['ground_truth'],
        'is_correct': result['is_correct'],
        'has_answer_probs': bool(answer_probs)
    }


def check_deepseek_file(filepath: str):
    """
    Check all results in a DeepSeek benchmark file.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"=" * 80)
    print(f"🔬 DEEPSEEK BENCHMARK LOGPROBS VALIDATION")
    print(f"=" * 80)
    print(f"File: {filepath.name}")
    print(f"Path: {filepath.parent}")
    print(f"Size: {filepath.stat().st_size / (1024*1024):.1f} MB")
    print(f"=" * 80)

    # Determine how to open
    if filepath.suffix == '.gz':
        open_func = lambda: gzip.open(filepath, 'rt')
    else:
        open_func = lambda: open(filepath, 'r')

    stats = []
    errors = []

    with open_func() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            try:
                data = json.loads(line.strip())

                # Handle both array format and single object format
                if isinstance(data, list):
                    # File contains arrays - process each item in the array
                    for item_idx, result in enumerate(data):
                        result_stats = check_single_result(result, idx * 1000 + item_idx)
                        stats.append(result_stats)
                else:
                    # File contains single objects
                    result = data
                    result_stats = check_single_result(result, idx)
                    stats.append(result_stats)
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                print(f"\n❌ ERROR at line {idx + 1}: {error_msg}")
                errors.append({'idx': idx, 'error': error_msg})
            except AssertionError as e:
                error_msg = str(e)
                print(f"\n❌ CRITICAL ERROR: {error_msg}")
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
        total_with_top = sum(s['num_with_top_logprobs'] for s in stats)
        correct = sum(1 for s in stats if s['is_correct'])

        print(f"\n📈 Statistics for Valid Results:")
        print(f"  - Total tokens: {total_tokens:,}")
        print(f"  - Tokens with top_logprobs: {total_with_top:,} ({100*total_with_top/total_tokens:.1f}%)")
        print(f"  - Average tokens per result: {total_tokens/len(stats):.1f}")
        print(f"  - Results with answer tokens: {sum(1 for s in stats if s['answer_tokens_found'] > 0)}/{len(stats)}")
        print(f"  - Results with answer probabilities: {sum(1 for s in stats if s['has_answer_probs'])}/{len(stats)}")
        print(f"  - Accuracy: {correct}/{len(stats)} ({100*correct/len(stats):.1f}%)")

    if errors:
        print(f"\n⚠️ ERRORS FOUND:")
        for error in errors[:5]:
            print(f"  - Result {error['idx'] + 1}: {error['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

        print(f"\n⚠️ Some results have issues but {len(stats)} results are valid.")
    else:
        print(f"\n✅ ALL RESULTS VALIDATED SUCCESSFULLY!")
        print(f"All {len(stats)} results have logprobs data preserved.")

    return stats, errors


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_deepseek_logprobs.py <results_file.jsonl.gz>")
        print("\nValidates DeepSeek benchmark results with logprobs.")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        stats, errors = check_deepseek_file(filepath)
        if not errors:
            print(f"\n🎉 Success! All results in '{Path(filepath).name}' are valid!")
        else:
            print(f"\n⚠️ Validation completed with {len(errors)} errors.")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()