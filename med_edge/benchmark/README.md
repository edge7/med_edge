# Medical Question Answering Benchmark Results

Benchmark results for reasoning models on medical QA datasets with full token-level logprobs for meta-learning.

---

## Models

### openai/gpt-oss-120b
- **Hardware:** NVIDIA H100 SXM5 80GB
- **Parameters:**
  - Temperature: 1.0
  - Max tokens: 32,768
  - Reasoning effort: medium
  - Top logprobs: 20

### deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- **Hardware:** NVIDIA H100 SXM5 80GB (same as above)
- **Parameters:**
  - Temperature: 0.6
  - Max tokens: 32,768
  - Reasoning effort: N/A (native reasoning)
  - Top logprobs: 20

---

## Datasets

### MedQA (bigbio/med_qa)
- Train: 10,178 questions
- Validation: 1,272 questions
- Test: 1,273 questions

### MedAgents (super-dainiu/medagents-benchmark)
Multiple configs: AfrimedQA, MMLU, MMLU-Pro, MedBullets, MedExQA, MedMCQA, MedXpertQA-R, MedXpertQA-U, PubMedQA

---

## Data Structure

Each benchmark produces two files per split:

### 1. CSV File (`*_train.csv`, `*_val.csv`, `*_test.csv`)
Processed results with extracted features.

**Key Columns:**
- **Metadata:** `dataset_name`, `split_name`, `question_index`, `sample_id`, `meta_info`
- **Question:** `question`, `option_a`, `option_b`, `option_c`, `option_d`, `option_e`
- **Answer:** `predicted_answer`, `ground_truth`, `is_correct`
- **Reasoning Features:** `reasoning_mean`, `reasoning_median`, `reasoning_std`, `reasoning_min`, `reasoning_max`, `reasoning_skewness`, `reasoning_kurtosis`, `reasoning_perplexity`, `reasoning_length`
- **Answer Confidence:** `answer_probability`, `answer_logprob`, `answer_entropy`, `answer_margin`, `prob_a`, `prob_b`, `prob_c`, `prob_d`, `prob_e`
- **Usage:** `prompt_tokens`, `completion_tokens`, `total_tokens`

### 2. JSON.GZ File (`*_train_raw.jsonl.gz`, etc.)
Complete raw data with full token-level logprobs.

**Structure:**
```json
{
  "dataset_name": "medqa",
  "split_name": "train",
  "question_index": 0,
  "sample_id": 8820869943274943983,
  "question": "Question text...",
  "options": {"a": "...", "b": "...", "c": "...", "d": "...", "e": "..."},
  "answer": "b",
  "ground_truth": "b",
  "is_correct": true,
  "reasoning_content": "Full reasoning text...",
  "meta_info": "step1",
  "temperature": 1.0,
  "max_tokens": 32768,
  "reasoning_effort": "medium",
  "usage": {
    "prompt_tokens": 272,
    "completion_tokens": 149,
    "total_tokens": 421
  },
  "logprobs": {
    "num_total_tokens": 149,
    "num_reasoning_tokens": 139,
    "num_json_structure_tokens": 6,
    "num_answer_tokens": 4,
    "answer_token_index": 145,
    "reasoning_end_index": 138,
    "content": [
      {
        "token": "Let",
        "logprob": -0.0001,
        "token_type": "reasoning",
        "top_logprobs": [
          {"token": "Let", "logprob": -0.0001},
          {"token": "Okay", "logprob": -9.2103},
          ...  // 20 total
        ]
      },
      ...
    ]
  }
}
```

**Key Features:**
- Token-level logprobs for every token
- 20 top alternatives per token
- Token classification: `reasoning`, `json_structure`, `answer`
- Full generation parameters for reproducibility

---

## Sample IDs

- **MedQA:** MD5 hash of question text (int64) - deterministic across runs
- **MedAgents:** Native `realidx` field (UUID string) - provided by dataset

Sample IDs allow joining results across different models for the same questions.

---

## Usage

### Load CSV
```python
import pandas as pd
df = pd.read_csv('openai_gpt-oss-120b_train.csv')
accuracy = df['is_correct'].mean()
```

### Load JSON with Logprobs
```python
import json, gzip
with gzip.open('openai_gpt-oss-120b_train_raw.jsonl.gz', 'rt') as f:
    data = json.load(f)

# Access token-level logprobs
logprobs = data[0]['logprobs']['content']
reasoning_tokens = [t for t in logprobs if t['token_type'] == 'reasoning']
```

### Join Multiple Models
```python
# Join by sample_id (same questions)
gpt = pd.read_csv('openai_gpt-oss-120b_train.csv')
deepseek = pd.read_csv('deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_train.csv')
merged = gpt.merge(deepseek, on='sample_id', suffixes=('_gpt', '_deepseek'))
```