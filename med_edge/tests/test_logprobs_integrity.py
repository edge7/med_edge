"""
Comprehensive tests for logprobs data integrity.

CRITICAL FOR RESEARCH: These tests ensure no data loss during serialization.

Run with: pytest test_logprobs_integrity.py -v
"""

import pytest
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_basic.vllm_native_request import serialize_logprobs
from benchmark.logprobs_validator import (
    validate_serialized_logprobs,
    validate_benchmark_result,
    extract_answer_probabilities,
    LogprobsValidationError,
)


# ==============================================================================
# Mock Data Creation
# ==============================================================================

class MockToken:
    """Mock token object that mimics OpenAI response structure."""
    def __init__(self, token, logprob, top_logprobs=None):
        self.token = token
        self.logprob = logprob
        self.top_logprobs = top_logprobs or []


class MockTopLogprob:
    """Mock top_logprob entry."""
    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


def create_mock_logprobs_data(answer='b', num_reasoning_tokens=50):
    """
    Create mock logprobs data that mimics real vLLM output.

    Structure:
    - reasoning_tokens (e.g., 50 tokens of reasoning)
    - JSON start token "{"
    - JSON structure tokens (e.g., '"answer"', ':', '"')
    - answer token (e.g., 'b')
    - JSON end tokens (e.g., '"', '}')
    """
    content = []

    # Add reasoning tokens
    for i in range(num_reasoning_tokens):
        top_logprobs = [
            MockTopLogprob(f"token_{j}", -0.1 * j)
            for j in range(20)
        ]
        content.append(MockToken(f"reasoning_{i}", -0.5, top_logprobs))

    # Add JSON start
    top_logprobs_json_start = [MockTopLogprob("{", -0.001)] + [
        MockTopLogprob(f"alt_{j}", -2.0) for j in range(19)
    ]
    content.append(MockToken("{", -0.001, top_logprobs_json_start))

    # Add JSON structure tokens
    for json_token in ['"answer"', ':', ' "']:
        top_logprobs = [MockTopLogprob(json_token, -0.001)] + [
            MockTopLogprob(f"alt_{j}", -3.0) for j in range(19)
        ]
        content.append(MockToken(json_token, -0.001, top_logprobs))

    # Add answer token with all options in top_logprobs
    # Make the selected answer most likely by setting its logprob highest
    answer_probs = {
        'a': -2.0,
        'b': -1.5,
        'c': -2.5,
        'd': -3.0,
        'e': -3.5,
    }
    # Set selected answer to be most likely
    answer_probs[answer] = -0.1

    answer_logprobs = [MockTopLogprob(letter, prob) for letter, prob in answer_probs.items()] + \
                      [MockTopLogprob(f"other_{j}", -5.0) for j in range(15)]

    # Add answer token
    selected_logprob = answer_probs[answer]
    content.append(MockToken(answer, selected_logprob, answer_logprobs))

    # Add JSON end tokens
    for json_token in ['"', '}']:
        top_logprobs = [MockTopLogprob(json_token, -0.001)] + [
            MockTopLogprob(f"alt_{j}", -4.0) for j in range(19)
        ]
        content.append(MockToken(json_token, -0.001, top_logprobs))

    return {'content': content}


# ==============================================================================
# Test serialize_logprobs
# ==============================================================================

def test_serialize_logprobs_basic():
    """Test basic serialization of logprobs."""
    logprobs_data = create_mock_logprobs_data(answer='b', num_reasoning_tokens=50)

    result = serialize_logprobs(logprobs_data, answer='b', reasoning_content='mock reasoning')

    assert result is not None
    assert 'content' in result
    assert 'answer_token_index' in result
    assert 'reasoning_end_index' in result
    assert len(result['content']) == len(logprobs_data['content'])


def test_serialize_logprobs_token_counts():
    """Test that token counts are correct."""
    logprobs_data = create_mock_logprobs_data(answer='c', num_reasoning_tokens=100)

    result = serialize_logprobs(logprobs_data, answer='c', reasoning_content='mock')

    # Should have 100 reasoning tokens
    assert result['num_reasoning_tokens'] == 100

    # Total should match input
    assert result['num_total_tokens'] == len(logprobs_data['content'])

    # Sum of parts should equal total
    total = (result['num_reasoning_tokens'] +
             result['num_json_structure_tokens'] +
             result['num_answer_tokens'])
    assert total == result['num_total_tokens']


def test_serialize_logprobs_preserves_top_logprobs():
    """CRITICAL: Test that all 20 top_logprobs are preserved for each token."""
    logprobs_data = create_mock_logprobs_data(answer='a', num_reasoning_tokens=30)

    result = serialize_logprobs(logprobs_data, answer='a', reasoning_content='mock')

    # Check every token has exactly 20 top_logprobs
    for idx, token in enumerate(result['content']):
        assert 'top_logprobs' in token, f"Token {idx} missing top_logprobs"
        assert len(token['top_logprobs']) == 20, \
            f"Token {idx} has {len(token['top_logprobs'])} top_logprobs, expected 20"

        # Check structure of each top_logprob
        for alt_idx, alt in enumerate(token['top_logprobs']):
            assert 'token' in alt, f"Token {idx} top_logprob {alt_idx} missing 'token'"
            assert 'logprob' in alt, f"Token {idx} top_logprob {alt_idx} missing 'logprob'"


def test_serialize_logprobs_answer_not_found_raises():
    """Test that serialize_logprobs raises if answer token not found."""
    logprobs_data = create_mock_logprobs_data(answer='b', num_reasoning_tokens=20)

    # Try to serialize with wrong answer
    with pytest.raises(AssertionError, match="Could not find answer token 'z'"):
        serialize_logprobs(logprobs_data, answer='z', reasoning_content='mock')


def test_serialize_logprobs_missing_top_logprobs_raises():
    """Test that serialize_logprobs raises if any token is missing top_logprobs."""
    # Create data with answer token 'a' present
    content = []
    for i in range(10):
        top_logprobs = [MockTopLogprob(f"token_{j}", -0.5) for j in range(20)]
        content.append(MockToken(f"reasoning_{i}", -0.5, top_logprobs))

    # Add answer with missing top_logprobs
    content.append(MockToken("a", -0.1, None))  # Missing top_logprobs!

    logprobs_data = {'content': content}

    with pytest.raises(AssertionError, match="missing top_logprobs"):
        serialize_logprobs(logprobs_data, answer='a', reasoning_content='mock')


def test_serialize_logprobs_wrong_number_top_logprobs_raises():
    """Test that serialize_logprobs raises if token has wrong number of top_logprobs."""
    # Create proper structure with answer token
    content = []
    for i in range(10):
        top_logprobs = [MockTopLogprob(f"token_{j}", -0.5) for j in range(20)]
        content.append(MockToken(f"reasoning_{i}", -0.5, top_logprobs))

    # Add answer token with wrong number of top_logprobs
    content.append(MockToken("a", -0.1, [MockTopLogprob(f"alt_{j}", -1.0) for j in range(15)]))  # Only 15!

    logprobs_data = {'content': content}

    with pytest.raises(AssertionError, match="has 15 top_logprobs, expected 20"):
        serialize_logprobs(logprobs_data, answer='a', reasoning_content='mock')


# ==============================================================================
# Test validate_serialized_logprobs
# ==============================================================================

def test_validate_serialized_logprobs_valid():
    """Test validation passes for valid serialized data."""
    logprobs_data = create_mock_logprobs_data(answer='d', num_reasoning_tokens=40)
    serialized = serialize_logprobs(logprobs_data, answer='d', reasoning_content='mock')

    # Should not raise
    validation_result = validate_serialized_logprobs(serialized, expected_answer='d')

    assert validation_result['valid'] is True
    assert validation_result['total_tokens'] > 0
    assert validation_result['reasoning_tokens'] == 40
    assert 0 <= validation_result['answer_probability'] <= 1


def test_validate_serialized_logprobs_none_raises():
    """Test validation raises if data is None."""
    with pytest.raises(LogprobsValidationError, match="Serialized data is None"):
        validate_serialized_logprobs(None, expected_answer='a')


def test_validate_serialized_logprobs_missing_fields_raises():
    """Test validation raises if required fields are missing."""
    incomplete_data = {
        'content': [],
        'answer_token_index': 0,
        # Missing other required fields
    }

    with pytest.raises(LogprobsValidationError, match="Missing required field"):
        validate_serialized_logprobs(incomplete_data, expected_answer='a')


def test_validate_serialized_logprobs_token_count_mismatch_raises():
    """Test validation raises if token counts don't match."""
    logprobs_data = create_mock_logprobs_data(answer='b', num_reasoning_tokens=30)
    serialized = serialize_logprobs(logprobs_data, answer='b', reasoning_content='mock')

    # Corrupt the token count
    serialized['num_total_tokens'] = 999

    with pytest.raises(LogprobsValidationError, match="Token count mismatch"):
        validate_serialized_logprobs(serialized, expected_answer='b')


def test_validate_serialized_logprobs_no_reasoning_tokens_raises():
    """Test validation raises if no reasoning tokens."""
    logprobs_data = create_mock_logprobs_data(answer='a', num_reasoning_tokens=0)

    # This should fail during serialization itself
    with pytest.raises(AssertionError, match="No reasoning tokens"):
        serialize_logprobs(logprobs_data, answer='a', reasoning_content='mock')


def test_validate_serialized_logprobs_too_few_reasoning_tokens_raises():
    """Test validation raises warning if suspiciously few reasoning tokens."""
    logprobs_data = create_mock_logprobs_data(answer='a', num_reasoning_tokens=5)
    serialized = serialize_logprobs(logprobs_data, answer='a', reasoning_content='mock')

    with pytest.raises(LogprobsValidationError, match="Suspiciously few reasoning tokens"):
        validate_serialized_logprobs(serialized, expected_answer='a')


# ==============================================================================
# Test extract_answer_probabilities
# ==============================================================================

def test_extract_answer_probabilities_all_options():
    """Test extracting probabilities for all valid options."""
    logprobs_data = create_mock_logprobs_data(answer='b', num_reasoning_tokens=30)
    serialized = serialize_logprobs(logprobs_data, answer='b', reasoning_content='mock')

    probs = extract_answer_probabilities(serialized, valid_options=['a', 'b', 'c', 'd', 'e'])

    # Should have all 5 options
    assert len(probs) == 5
    assert all(opt in probs for opt in ['a', 'b', 'c', 'd', 'e'])

    # All probabilities should be in [0, 1]
    for opt, prob in probs.items():
        assert 0 <= prob <= 1, f"Probability for {opt} is {prob}, not in [0, 1]"

    # Selected answer should have highest probability
    assert probs['b'] == max(probs.values())


def test_extract_answer_probabilities_subset():
    """Test extracting probabilities for subset of options (e.g., 3-option question)."""
    logprobs_data = create_mock_logprobs_data(answer='b', num_reasoning_tokens=30)
    serialized = serialize_logprobs(logprobs_data, answer='b', reasoning_content='mock')

    probs = extract_answer_probabilities(serialized, valid_options=['a', 'b', 'c'])

    # Should have only 3 options
    assert len(probs) == 3
    assert all(opt in probs for opt in ['a', 'b', 'c'])


def test_extract_answer_probabilities_missing_option_raises():
    """Test that extraction raises if valid option is not in top 20."""
    # Create data where only a, b, c are in top_logprobs (no d, e)
    content = []
    for i in range(20):
        top_logprobs = [MockTopLogprob("reasoning", -0.5)] * 20
        content.append(MockToken(f"tok_{i}", -0.5, top_logprobs))

    # Add answer token with only 3 options in top_logprobs
    answer_top_logprobs = [
        MockTopLogprob('a', -1.0),
        MockTopLogprob('b', -0.1),
        MockTopLogprob('c', -1.5),
    ] + [MockTopLogprob(f"other_{j}", -5.0) for j in range(17)]

    content.append(MockToken("{", -0.001, answer_top_logprobs[:20]))
    for tok in ['"answer"', ':', ' "']:
        content.append(MockToken(tok, -0.001, answer_top_logprobs[:20]))

    content.append(MockToken('b', -0.1, answer_top_logprobs[:20]))

    for tok in ['"', '}']:
        content.append(MockToken(tok, -0.001, answer_top_logprobs[:20]))

    logprobs_data = {'content': content}
    serialized = serialize_logprobs(logprobs_data, answer='b', reasoning_content='mock')

    # Try to extract with d and e (which aren't in top_logprobs)
    with pytest.raises(LogprobsValidationError, match="Cannot extract probabilities"):
        extract_answer_probabilities(serialized, valid_options=['a', 'b', 'c', 'd', 'e'])


# ==============================================================================
# Test validate_benchmark_result
# ==============================================================================

def test_validate_benchmark_result_complete():
    """Test validation of complete benchmark result."""
    logprobs_data = create_mock_logprobs_data(answer='c', num_reasoning_tokens=50)
    serialized = serialize_logprobs(logprobs_data, answer='c', reasoning_content='mock reasoning')

    result = {
        'answer': 'c',
        'ground_truth': 'c',
        'is_correct': True,
        'usage': {
            'prompt_tokens': 100,
            'completion_tokens': 60,
            'total_tokens': 160,
        },
        'reasoning_content': 'mock reasoning content',
        'finish_reason': 'stop',
        'temperature': 0.6,
        'max_tokens': 32768,
        'logprobs': serialized,
    }

    # Should not raise
    is_valid = validate_benchmark_result(result)
    assert is_valid is True


def test_validate_benchmark_result_missing_fields_raises():
    """Test that validation raises if required fields are missing."""
    incomplete_result = {
        'answer': 'a',
        'ground_truth': 'a',
        # Missing other required fields
    }

    with pytest.raises(LogprobsValidationError, match="missing required field"):
        validate_benchmark_result(incomplete_result)


def test_validate_benchmark_result_missing_logprobs_raises():
    """CRITICAL: Test that validation raises if logprobs is None."""
    result = {
        'answer': 'a',
        'ground_truth': 'a',
        'is_correct': True,
        'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
        'reasoning_content': 'reasoning',
        'finish_reason': 'stop',
        'temperature': 0.6,
        'max_tokens': 1000,
        'logprobs': None,  # MISSING!
    }

    with pytest.raises(LogprobsValidationError, match="Logprobs is None"):
        validate_benchmark_result(result)


def test_validate_benchmark_result_invalid_usage_raises():
    """Test that validation raises if usage fields are invalid."""
    logprobs_data = create_mock_logprobs_data(answer='a', num_reasoning_tokens=20)
    serialized = serialize_logprobs(logprobs_data, answer='a', reasoning_content='mock')

    result = {
        'answer': 'a',
        'ground_truth': 'a',
        'is_correct': True,
        'usage': {
            'prompt_tokens': -10,  # Negative!
            'completion_tokens': 10,
            'total_tokens': 20,
        },
        'reasoning_content': 'reasoning',
        'finish_reason': 'stop',
        'temperature': 0.6,
        'max_tokens': 1000,
        'logprobs': serialized,
    }

    with pytest.raises(LogprobsValidationError, match="must be non-negative integer"):
        validate_benchmark_result(result)


# ==============================================================================
# Integration Test: Round-trip
# ==============================================================================

def test_round_trip_serialization_and_validation():
    """
    Integration test: Serialize mock data and validate it can be used for research.

    This simulates the full pipeline:
    1. Get logprobs from vLLM
    2. Serialize them
    3. Save to JSON
    4. Load from JSON
    5. Validate all data is intact
    6. Extract answer probabilities
    """
    # Step 1 & 2: Create and serialize
    logprobs_data = create_mock_logprobs_data(answer='d', num_reasoning_tokens=100)
    serialized = serialize_logprobs(logprobs_data, answer='d', reasoning_content='Long reasoning...')

    # Step 3 & 4: Simulate JSON round-trip (serialization test)
    import json
    json_str = json.dumps(serialized)
    loaded = json.loads(json_str)

    # Step 5: Validate loaded data
    validation_result = validate_serialized_logprobs(loaded, expected_answer='d')
    assert validation_result['valid'] is True
    assert validation_result['reasoning_tokens'] == 100

    # Step 6: Extract answer probabilities
    probs = extract_answer_probabilities(loaded, valid_options=['a', 'b', 'c', 'd', 'e'])
    assert len(probs) == 5
    assert probs['d'] == max(probs.values())  # d should be most likely

    # Verify we can compute answer confidence
    answer_prob = probs['d']
    answer_logprob = validation_result['answer_logprob']
    assert abs(answer_prob - math.exp(answer_logprob)) < 1e-6

    print(f"\n✅ Round-trip test passed!")
    print(f"   Total tokens: {validation_result['total_tokens']}")
    print(f"   Reasoning tokens: {validation_result['reasoning_tokens']}")
    print(f"   Answer probability: {answer_prob:.4f}")
    print(f"   Answer log prob: {answer_logprob:.4f}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
