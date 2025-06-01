# AgentX OpenProbe

## 🚀 Getting Started  

## Features

- **Automated Planning**: Breaks down complex queries into multiple search steps
- **Adaptive Replanning**: Analyzes search results and revises the search strategy when initial plans are insufficient (limited to 2 replans)
- **Reflection**: Provides reasoning about why previous plans failed and how to improve them
- **Web Search Integration**: Seamlessly integrates with search APIs to gather information

### Current Architecture


#### How it Works
1. The system analyzes the user's question
2. It creates a search plan with multiple sub-queries
3. It executes searches based on the plan
4. If results are insufficient, it can replan with improved queries (up to 2 times)
5. Finally, it synthesizes all information into a comprehensive answer

## How to Run

### Set Up API Keys

Create a `.env` file under the `openprobe` directory and set each API key properly

```
GOOGLE_API_KEY=your_gemini_api_key
LAMBDA_API_KEY=your_lambda_api_key
WEB_SEARCH_API_KEY=your_serper_dev_api_key
JINA_API_KEY=your_jina_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

### Installation

Run the following command to complete setup

```bash
cd openprobe
pip install -e .
crawl4ai-setup
crawl4ai-doctor
```

### Run with Single Question

```bash
python test_deepsearch.py
```

### Run Evaluation with FRAMES

Run evaluation of FRAMES subset with the following command:

```bash
python evals/eval_tasks.py \
    --eval-tasks ./evals/datasets/frames_custom_set.csv \
    --parallel-workers 8
```

After the evaluation completes, a jsonl file will store the result. The file will be located under the `output` directory.

### Run Auto Grading

Run LLM auto grading with the following command:

```bash
python evals/autograde_df.py \
    PATH_TO_RESULT_JSONL_FILE \
    --provider mistral \
    --num_cpus 2
```

After the grading completes, the grading result will be added to the input jsonl file.

### Compute Accuracy

Finally, run the following command get accuracy on the test set:

```bash
python evals/accuracy.py \
    PATH_TO_GRADED_JSONL_FILE
```
