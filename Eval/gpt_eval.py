import os
import multiprocessing

# Ensure multiprocessing works correctly
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from tqdm import tqdm
import sys

sys.path.append("../data/")
sys.path.append("../")

import re
import time
import pandas as pd
import tiktoken
from openai import OpenAI, AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import setproctitle

# Set process title
setproctitle.setproctitle("gpt-judge")

# Initialize tokenizer
encoding = tiktoken.encoding_for_model('gpt-4o')

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question,
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something based on some prior conversations or a passage.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it expresses the same key meaning as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

If the answer is vague or not specific enough, it should be considered WRONG. For example, if the gold answer is "A shell necklace" and the generated answer is "something from Hawaii", it should be considered WRONG.

The gold answer may contain multiple acceptable answers separated by "or". If the generated answer matches any one of these, it should be considered CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
"""


def evaluate_llm_judge(question, gold_answer, generated_answer, use_azure=False):
    """
    Calls the LLM to judge the correctness of the generated answer.
    """
    if not use_azure:
        model_name = "gpt-5-nano"
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', "your_openai_api_key_here"))
    else:
        model_name = "gpt-5-mini"
        api_key = os.environ.get('AZURE_API_KEY', "your_azure_api_key_here")
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            api_key=api_key,
            azure_endpoint="",
            azure_deployment=model_name
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        temperature=1.0,
    )

    label = response.choices[0].message.content
    result = 0 if "wrong" in label.lower() else 1

    return result


def extract_short_response(response, is_thinking_model=True, max_answer_len=4000):
    """
    Extracts the final response content.
    For thinking models, it extracts the content after the </think> tag.
    It returns an empty string if:
    1. The response is not a string.
    2. The thinking tag is missing (for thinking models).
    3. The extracted answer exceeds max_answer_len.

    Args:
        response: The raw string response from the model.
        is_thinking_model: Boolean indicating if the model uses thinking tags.
        max_answer_len: Maximum allowed token length for the answer.

    Returns:
        The extracted short response string or empty string.
    """
    if not isinstance(response, str):
        return ""

    final_answer = response

    if is_thinking_model:
        if "</think>" in response:
            final_answer = response.split("</think>")[-1].strip()
        else:
            # Tag missing, treat as invalid
            return ""

    # Check length using tokenizer
    if len(encoding.encode(final_answer)) > max_answer_len:
        return ""

    return final_answer


def extract_answer_letter(response):
    """
    Helper: Extracts the multiple choice answer (A, B, C, D) from the response.
    """
    response = response.replace('*', '')
    match = re.findall(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match[-1]
    else:
        match = re.findall(r'The correct answer is ([A-D])', response)
        if match:
            return match[-1]
        else:
            return ""


# --- Parallel Evaluation Functions ---

def evaluate_gpt_row(row, use_azure=False):
    """
    Evaluation logic for default dataset using GPT as judge.
    Assumes 'short_response' is already populated in the row.
    """
    row_data = row[1]
    question = row_data["question"]
    gold_answer = row_data["answer"]
    final_answer = row_data["short_response"]

    # If extracted answer is empty (due to length or missing tag), fail immediately
    if not final_answer:
        return 0

    # Retry logic if generation fails
    max_retries = 3
    label = 0
    for attempt in range(max_retries):
        try:
            label = evaluate_llm_judge(question, gold_answer, final_answer, use_azure=use_azure)
            break
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(60)
            else:
                print("Max retries reached. Returning 0.")
                return 0
    return label


def evaluate_longbenchv2_row(row):
    """
    Evaluation logic for LongBench-v2 dataset.
    Extracts choice letter from 'short_response' and compares with 'answer'.
    """
    row_data = row[1]
    gold_answer = row_data["answer"]
    short_response = row_data["short_response"]

    predicted_answer = extract_answer_letter(str(short_response))

    return 1 if predicted_answer.strip().lower() == gold_answer.strip().lower() else 0


def evaluate_ruler_row(row):
    """
    Evaluation logic for Ruler dataset.
    Checks if all items in the gold answer list are present in 'short_response'.
    """
    row_data = row[1]
    gold_answer_list = list(row_data["answer"])
    short_response = row_data["short_response"]

    if not short_response:
        return 0

    short_response_lower = short_response.lower()

    for answer_item in gold_answer_list:
        if str(answer_item).lower() not in short_response_lower:
            return 0
    return 1


# --- Main Eval Function ---

def eval_df(df, dataset_type="default", is_thinking_model=True, num_workers=50, use_azure=False):
    """
    Main evaluation function used to evaluate a dataframe of model outputs.

    Args:
        df: Pandas DataFrame containing model outputs.
        dataset_type: Type of dataset ("longbench-v2", "ruler", or others for GPT eval).
        is_thinking_model: Whether the model uses thinking tags (e.g. </think>).
        num_workers: Number of threads for evaluation.
        use_azure: Whether to use Azure OpenAI API (for GPT eval).
    """

    if "prompt_len" not in df.columns:
        df['prompt_len'] = df['prompt'].apply(lambda x: len(encoding.encode(x)))

    # Explode 'response' column if it's a list (for num_generations > 1)
    num_generations = 1
    if not df.empty and not isinstance(df['response'].iloc[0], str):
        try:
            num_generations = len(df['response'].iloc[0])
            df = df.explode('response').reset_index(drop=True)
            print("Expanded df to {} rows due to {} generations per prompt.".format(len(df), num_generations))
        except:
            pass

    # 1. Pre-calculate short_response for ALL dataset types
    # This centralizes the extraction and max_len logic
    df['short_response'] = df['response'].apply(
        lambda x: extract_short_response(x, is_thinking_model, max_answer_len=4000)
    )

    # 2. Select the appropriate evaluation function based on dataset_type
    if dataset_type == "longbench-v2":
        process_row_func = evaluate_longbenchv2_row
    elif dataset_type == "ruler":
        process_row_func = evaluate_ruler_row
    else:
        # Default to GPT evaluation
        process_row_func = partial(evaluate_gpt_row, use_azure=use_azure)

    # 3. Execute evaluation using ThreadPoolExecutor for unified parallelism
    # Note: Even for CPU-bound tasks (longbench/ruler), using ThreadPoolExecutor allows
    # a unified code structure as requested, though process_map or direct apply could be faster for them.
    print(f"Starting evaluation for dataset_type: {dataset_type} with {num_workers} workers.")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_row_func, df.iterrows()), total=len(df), mininterval=5))

    df['judge'] = results
    #set judge as int
    df['judge'] = df['judge'].apply(int).astype(object)

    # --- Metrics Calculation ---

    # 1. Calculate overall accuracy
    accuracy = float(round(df['judge'].mean(), 3))

    # 2. Calculate accuracy by question_type
    accuracy_by_type = None
    if "question_type" in df.columns:
        accuracy_by_type = df.groupby('question_type')['judge'].mean().to_dict()
        accuracy_by_type = {k: float(round(v, 3)) for k, v in accuracy_by_type.items()}

    print("Overall accuracy:", accuracy)

    # 3. Calculate accuracy by prompt_length bin
    bins = [0, 16000, 32000, 64000, 128000, float('inf')]
    labels = ['0-16k', '16-32k', '32-64k', '64-128k', '128k+']
    df['prompt_length_bin'] = pd.cut(df['prompt_len'], bins=bins, labels=labels, right=False)

    accuracy_by_prompt_length = df.groupby('prompt_length_bin', observed=False)['judge'].mean().to_dict()
    accuracy_by_prompt_length = {k: float(round(v, 3)) if not pd.isna(v) else 0 for k, v in accuracy_by_prompt_length.items()}

    # 4. Calculate 'think' tag missing ratio
    no_think_tag_ratio = None
    if is_thinking_model:
        # Handle non-string responses safely
        no_think_tag_ratio = float(round((df['response'].apply(lambda x: "</think>" not in str(x))).mean(), 3))

    # 5. Calculate response lengths
    df['response_length'] = df['response'].apply(lambda x: len(encoding.encode(str(x))) if isinstance(x, str) else 0)
    response_mean_len = float(round(df['response_length'].mean(), 1))

    correct_df = df[df['judge'] == 1]
    response_mean_len_judge_1 = float(round(correct_df['response_length'].mean(), 1)) if not correct_df.empty else 0.0

    # 6. Calculate Pass@k (if num_generations > 1)
    accuracy_passk = accuracy
    if 'original_index' in df.columns:
        # Re-aggregate to calculate pass@k
        df_agg = df.groupby('original_index').agg({
            'question': 'first',
            'answer': 'first',
            'response': list,
            'judge': list,
            'question_type': 'first' if 'question_type' in df.columns else (lambda x: None),
            'response_length': list
        }).reset_index(drop=True)

        df_agg['judge_agg'] = df_agg['judge'].apply(lambda x: max(x))
        accuracy_passk = float(round(df_agg['judge_agg'].mean(), 3))

    metrics = {
        "overall_accuracy": accuracy,
        "accuracy_passk": accuracy_passk,
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_prompt_length": accuracy_by_prompt_length,
        "no_think_tag_ratio": no_think_tag_ratio,
        "avg_response_len": response_mean_len,
        "avg_correct_response_len": response_mean_len_judge_1,
        "num_samples": len(df)
    }

    return df, metrics


if __name__ == '__main__':
    # Example usage
    df_list = [
        "//share/yyj/llm_as_memory/sliding-window-attention-adaptation/Eval/eval_output/ruler_niah_multiquery_128k/Qwen3-4B-Thinking-2507__vllm.parquet",
    ]
    for dataset_path in df_list:
        dataset_name = dataset_path.split("/")[-1].replace(".parquet", "")
        is_thinking_model = "thinking" in dataset_path.lower()
        print("Is thinking model:", is_thinking_model)

        print("Evaluating dataset:", dataset_path)
        # Assuming we can read the file
        if os.path.exists(dataset_path):
            df = pd.read_parquet(dataset_path)

            # Determine dataset_type based on name or other logic
            current_dataset_type = "default"
            if "longbench" in dataset_path.lower():
                current_dataset_type = "longbench-v2"
            elif "ruler" in dataset_path.lower():
                current_dataset_type = "ruler"

            # Evaluate
            df, metric = eval_df(
                df,
                is_thinking_model=is_thinking_model,
                num_workers=10,
                use_azure=True,
                dataset_type=current_dataset_type
            )

            # Save results
            df.to_parquet(dataset_path)
            print("Metrics:", metric)
            print("Results saved to", dataset_path)
        else:
            print(f"File not found: {dataset_path}")