from collections import namedtuple
from datasets import load_dataset


DatasetSplits = namedtuple('DatasetSplits', ['train', 'val', 'test'])


def get_med_qa_dataset():
    """
    Load the bigbio/med_qa dataset from Hugging Face with all splits.

    Returns:
        DatasetSplits: Named tuple containing train, val, and test datasets
    """
    train = load_dataset('bigbio/med_qa', split='train')
    val = load_dataset('bigbio/med_qa', split='validation')
    test = load_dataset('bigbio/med_qa', split='test')

    return DatasetSplits(train=train, val=val, test=test)


def get_medagents_benchmark_dataset():
    """
    Load the super-dainiu/medagents-benchmark dataset from Hugging Face with all splits.

    Returns:
        DatasetSplits: Named tuple containing train, val, and test datasets
    """
    train = load_dataset('super-dainiu/medagents-benchmark', split='train')
    val = load_dataset('super-dainiu/medagents-benchmark', split='validation')
    test = load_dataset('super-dainiu/medagents-benchmark', split='test')

    return DatasetSplits(train=train, val=val, test=test)