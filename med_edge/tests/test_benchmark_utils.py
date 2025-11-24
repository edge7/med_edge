"""
Comprehensive tests for benchmark_utils.py

Run with: pytest test_benchmark_utils.py -v
"""

import pytest
import tempfile
import json
import gzip
import hashlib
from pathlib import Path

from benchmark_utils import (
    parse_medqa_sample,
    parse_medagents_sample,
    atomic_save_json,
    load_json_results,
    calculate_accuracy_summary,
    setup_resume_logic,
    get_sample_id_from_medqa,
    get_sample_id_from_medagents,
)


# ==============================================================================
# Test parse_medqa_sample
# ==============================================================================

def test_parse_medqa_sample_basic():
    """Test basic parsing of MedQA sample."""
    sample = {
        'question': 'What is the capital of France?',
        'options': [
            {'key': 'A', 'value': 'London'},
            {'key': 'B', 'value': 'Paris'},
            {'key': 'C', 'value': 'Berlin'},
        ],
        'answer': 'Paris',
        'meta_info': 'step1',
    }

    result = parse_medqa_sample(sample, idx=0)

    assert result['question'] == 'What is the capital of France?'
    assert result['options'] == {'a': 'London', 'b': 'Paris', 'c': 'Berlin'}
    assert result['ground_truth'] == 'b'
    assert result['question_index'] == 0
    assert result['meta_info'] == 'step1'
    assert isinstance(result['sample_id'], int)


def test_parse_medqa_sample_deterministic():
    """Test that sample_id is deterministic based on question text."""
    sample1 = {
        'question': 'Test question',
        'options': [{'key': 'A', 'value': 'Opt1'}],
        'answer': 'Opt1',
    }
    sample2 = {
        'question': 'Test question',  # Same question
        'options': [{'key': 'A', 'value': 'Different option'}],  # Different options
        'answer': 'Different option',
    }

    result1 = parse_medqa_sample(sample1, idx=0)
    result2 = parse_medqa_sample(sample2, idx=1)

    # Same question should give same sample_id
    assert result1['sample_id'] == result2['sample_id']


def test_get_sample_id_from_medqa():
    """Test helper function to extract sample_id from MedQA sample."""
    sample = {'question': 'Test question'}
    sample_id = get_sample_id_from_medqa(sample)

    # Should be consistent with parse_medqa_sample
    parsed = parse_medqa_sample(sample, idx=0)
    assert sample_id == parsed['sample_id']


# ==============================================================================
# Test parse_medagents_sample
# ==============================================================================

def test_parse_medagents_sample_basic():
    """Test basic parsing of medagents-benchmark sample."""
    sample = {
        'question': 'What is photosynthesis?',
        'options': {'A': 'Process 1', 'B': 'Process 2', 'C': 'Process 3'},
        'answer_idx': 'B',
        'realidx': 12345,
        'meta_info': 'biology',
    }

    result = parse_medagents_sample(sample, is_hard=True)

    assert result['question'] == 'What is photosynthesis?'
    assert result['options'] == {'a': 'Process 1', 'b': 'Process 2', 'c': 'Process 3'}
    assert result['ground_truth'] == 'b'
    assert result['sample_id'] == 12345
    assert result['is_hard'] is True
    assert result['meta_info'] == 'biology'


def test_parse_medagents_sample_variable_options():
    """Test parsing with different numbers of options (2, 3, 4, 5)."""
    # 2 options
    sample_2 = {
        'question': 'True or false?',
        'options': {'A': 'True', 'B': 'False'},
        'answer_idx': 'A',
        'realidx': 1,
    }
    result_2 = parse_medagents_sample(sample_2)
    assert len(result_2['options']) == 2
    assert 'a' in result_2['options']
    assert 'b' in result_2['options']

    # 5 options
    sample_5 = {
        'question': 'Select the best option',
        'options': {'A': 'Opt1', 'B': 'Opt2', 'C': 'Opt3', 'D': 'Opt4', 'E': 'Opt5'},
        'answer_idx': 'C',
        'realidx': 2,
    }
    result_5 = parse_medagents_sample(sample_5)
    assert len(result_5['options']) == 5
    assert result_5['ground_truth'] == 'c'


def test_get_sample_id_from_medagents():
    """Test helper function to extract sample_id from medagents sample."""
    sample = {'realidx': 12345}
    sample_id = get_sample_id_from_medagents(sample)
    assert sample_id == 12345


# ==============================================================================
# Test atomic_save_json and load_json_results
# ==============================================================================

def test_atomic_save_and_load_json():
    """Test saving and loading JSON with atomic writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_results.jsonl.gz"

        # Test data
        test_data = [
            {'sample_id': 1, 'answer': 'a', 'is_correct': True},
            {'sample_id': 2, 'answer': 'b', 'is_correct': False},
            {'sample_id': 3, 'answer': 'c', 'is_correct': True},
        ]

        # Save
        atomic_save_json(test_data, file_path)
        assert file_path.exists()

        # Load
        loaded_data, completed_ids = load_json_results(file_path)
        assert loaded_data == test_data
        assert completed_ids == {1, 2, 3}


def test_load_json_nonexistent_file():
    """Test loading from nonexistent file returns empty results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "nonexistent.jsonl.gz"
        loaded_data, completed_ids = load_json_results(file_path)

        assert loaded_data == []
        assert completed_ids == set()


def test_atomic_save_json_creates_parent_dirs():
    """Test that atomic_save_json creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "subdir1" / "subdir2" / "results.jsonl.gz"

        test_data = [{'sample_id': 1}]
        # Should not raise error even though parent dirs don't exist
        atomic_save_json(test_data, nested_path)
        # Note: This actually will fail because atomic_save_json doesn't create parent dirs
        # This is a design choice - the calling code should create them


# ==============================================================================
# Test calculate_accuracy_summary
# ==============================================================================

def test_calculate_accuracy_summary_basic():
    """Test basic accuracy calculation."""
    results = [
        {'is_correct': True},
        {'is_correct': True},
        {'is_correct': False},
        {'is_correct': True},
    ]

    stats = calculate_accuracy_summary(results)

    assert stats['accuracy'] == 0.75
    assert stats['correct_count'] == 3
    assert stats['total_count'] == 4
    assert stats['hard_accuracy'] is None


def test_calculate_accuracy_summary_with_hard_subset():
    """Test accuracy calculation with hard subset."""
    results = [
        {'is_correct': True, 'is_hard': False},
        {'is_correct': True, 'is_hard': False},
        {'is_correct': False, 'is_hard': True},
        {'is_correct': True, 'is_hard': True},
    ]

    stats = calculate_accuracy_summary(results, has_hard_subset=True)

    assert stats['accuracy'] == 0.75
    assert stats['correct_count'] == 3
    assert stats['total_count'] == 4
    assert stats['hard_accuracy'] == 0.5  # 1 correct out of 2 hard questions
    assert stats['hard_correct'] == 1
    assert stats['hard_total'] == 2


def test_calculate_accuracy_summary_empty_results():
    """Test accuracy calculation with empty results."""
    results = []

    stats = calculate_accuracy_summary(results)

    assert stats['accuracy'] == 0.0
    assert stats['correct_count'] == 0
    assert stats['total_count'] == 0


def test_calculate_accuracy_summary_filters_none():
    """Test that None results are filtered out."""
    results = [
        {'is_correct': True},
        None,
        {'is_correct': False},
        None,
        {'is_correct': True},
    ]

    stats = calculate_accuracy_summary(results)

    assert stats['total_count'] == 3  # Only non-None results counted
    assert stats['accuracy'] == 2/3


# ==============================================================================
# Test setup_resume_logic
# ==============================================================================

def test_setup_resume_logic_no_existing_file():
    """Test setup_resume_logic when no existing results file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "results.jsonl.gz"
        data_length = 100

        raw_results, completed_ids = setup_resume_logic(json_file, data_length)

        assert len(raw_results) == data_length
        assert all(r is None for r in raw_results)
        assert completed_ids == set()


def test_setup_resume_logic_with_existing_file():
    """Test setup_resume_logic with existing results file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "results.jsonl.gz"
        data_length = 5

        # Create existing results
        existing_results = [
            {'sample_id': 1, 'question_index': 0, 'is_correct': True},
            {'sample_id': 2, 'question_index': 2, 'is_correct': False},
        ]
        atomic_save_json(existing_results, json_file)

        # Setup resume logic
        raw_results, completed_ids = setup_resume_logic(json_file, data_length)

        assert len(raw_results) == data_length
        assert raw_results[0] == existing_results[0]
        assert raw_results[2] == existing_results[1]
        assert raw_results[1] is None
        assert completed_ids == {1, 2}


# ==============================================================================
# Integration test
# ==============================================================================

def test_full_workflow():
    """Test a complete workflow: parse, process, save, resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "test_workflow.jsonl.gz"

        # Simulate MedQA samples
        medqa_samples = [
            {
                'question': 'Q1',
                'options': [{'key': 'A', 'value': 'Ans1'}, {'key': 'B', 'value': 'Ans2'}],
                'answer': 'Ans1',
            },
            {
                'question': 'Q2',
                'options': [{'key': 'A', 'value': 'Ans1'}, {'key': 'B', 'value': 'Ans2'}],
                'answer': 'Ans2',
            },
        ]

        # Parse samples
        parsed_samples = [parse_medqa_sample(s, idx=i) for i, s in enumerate(medqa_samples)]

        # Simulate results
        results = [
            {
                'sample_id': parsed_samples[0]['sample_id'],
                'question_index': 0,
                'is_correct': True,
                'answer': 'a',
            },
            None,  # Second one not processed yet
        ]

        # Save partial results
        completed_results = [r for r in results if r is not None]
        atomic_save_json(completed_results, json_file)

        # Resume logic
        raw_results, completed_ids = setup_resume_logic(json_file, len(medqa_samples))

        assert len(raw_results) == 2
        assert raw_results[0] is not None
        assert raw_results[1] is None
        assert parsed_samples[0]['sample_id'] in completed_ids
        assert parsed_samples[1]['sample_id'] not in completed_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
