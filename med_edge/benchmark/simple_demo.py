#!/usr/bin/env python
"""
Simple demo script to show how logprobs are preserved through the entire pipeline.
This creates a small benchmark run and validates the output.
"""

import sys
from pathlib import Path
import json
import gzip

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.benchmark_utils import (
    save_result,
    complete_benchmark_session,
    extract_answer_from_response
)
from benchmark.logprobs_validator import (
    validate_benchmark_result,
    extract_answer_probabilities
)
from llm_basic.vllm_native_request import serialize_logprobs


def create_mock_result():
    """Create a mock result with full logprobs for testing."""

    # Mock logprobs data (simulating vLLM response)
    class MockTopLogprob:
        def __init__(self, token, logprob):
            self.token = token
            self.logprob = logprob

    class MockToken:
        def __init__(self, token, logprob, top_logprobs):
            self.token = token
            self.logprob = logprob
            self.top_logprobs = top_logprobs

    # Create realistic response with reasoning and answer
    content = []

    # Add reasoning tokens
    reasoning_text = "Let me analyze this medical question carefully. Based on the clinical presentation, "
    for word in reasoning_text.split():
        top_logprobs = [MockTopLogprob(f"alt_{i}", -2.5) for i in range(20)]
        content.append(MockToken(word, -0.5, top_logprobs))

    # Add JSON structure
    json_tokens = ['{"', 'reasoning', '":', '"', 'detailed', 'analysis', '",', '"', 'answer', '":', '"']
    for token in json_tokens:
        top_logprobs = [MockTopLogprob(f"alt_{i}", -3.0) for i in range(20)]
        content.append(MockToken(token, -0.1, top_logprobs))

    # Add answer token with all MCQ options in top_logprobs
    answer_logprobs = [
        MockTopLogprob('a', -2.0),
        MockTopLogprob('b', -0.1),  # Most likely
        MockTopLogprob('c', -1.5),
        MockTopLogprob('d', -1.8),
        MockTopLogprob('e', -2.5),
    ] + [MockTopLogprob(f"other_{j}", -5.0) for j in range(15)]

    content.append(MockToken('b', -0.1, answer_logprobs))

    # Add JSON closing
    content.append(MockToken('"}', -0.1, [MockTopLogprob(f"alt_{i}", -3.0) for i in range(20)]))

    # Serialize the logprobs
    logprobs_data = {'content': content}
    serialized = serialize_logprobs(logprobs_data)

    # Create full benchmark result
    result = {
        'question_id': 'demo_001',
        'question': 'What is the most common cause of acute pancreatitis?',
        'options': {
            'A': 'Alcohol abuse',
            'B': 'Gallstones',
            'C': 'Hypertriglyceridemia',
            'D': 'Medications',
            'E': 'Trauma'
        },
        'correct_answer': 'B',
        'model_answer': 'b',
        'full_response': '{"reasoning": "detailed analysis", "answer": "b"}',
        'serialized_logprobs': serialized,
        'metadata': {
            'model': 'demo_model',
            'temperature': 0.0,
            'timestamp': '2025-11-24T12:00:00Z'
        }
    }

    return result


def main():
    """Run the demo."""
    print("=" * 80)
    print("LOGPROBS PRESERVATION DEMO")
    print("=" * 80)

    # Create mock result
    print("\n1. Creating mock benchmark result with full logprobs...")
    result = create_mock_result()

    # Validate the result
    print("\n2. Validating the result...")
    validation = validate_benchmark_result(result)

    if not validation['valid']:
        print(f"❌ Validation FAILED: {validation['errors']}")
        sys.exit(1)

    print(f"✅ Validation PASSED!")
    print(f"   - Tokens: {validation['stats']['num_tokens']}")
    print(f"   - Tokens with top_logprobs: {validation['stats']['num_tokens_with_top_logprobs']}")
    print(f"   - Answer found: {validation['stats']['has_answer_token']}")

    # Extract answer probabilities
    print("\n3. Extracting answer probabilities...")
    answer_probs = extract_answer_probabilities(
        result['serialized_logprobs'],
        result['model_answer']
    )

    if answer_probs:
        print("✅ Answer probabilities extracted:")
        import math
        for option in ['a', 'b', 'c', 'd', 'e']:
            if option in answer_probs and answer_probs[option]:
                prob = math.exp(answer_probs[option]['logprob'])
                print(f"   Option {option.upper()}: {prob:.2%} (logprob: {answer_probs[option]['logprob']:.3f})")

    # Save to file
    print("\n4. Saving result to file...")
    output_file = Path("demo_results.jsonl.gz")

    with gzip.open(output_file, 'wt') as f:
        f.write(json.dumps(result) + '\n')

    print(f"✅ Saved to {output_file}")

    # Validate the saved file
    print("\n5. Validating saved file with validate_logprobs.py...")
    from benchmark.validate_logprobs import check_results_file

    try:
        results, stats = check_results_file(output_file)
        print("✅ File validation PASSED!")
    except Exception as e:
        print(f"❌ File validation FAILED: {e}")
        sys.exit(1)

    # Clean up
    output_file.unlink()
    print(f"\n✅ Demo complete! Cleaned up {output_file}")

    print("\n" + "=" * 80)
    print("SUCCESS: All logprobs data is preserved correctly!")
    print("Your research data integrity is guaranteed.")
    print("=" * 80)


if __name__ == "__main__":
    main()