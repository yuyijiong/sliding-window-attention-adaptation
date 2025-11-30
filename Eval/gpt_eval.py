import os

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from tqdm import tqdm
import sys
sys.path.append("../data/")
sys.path.append("../")

import re

import time

import pandas as pd
import tiktoken
encoding = tiktoken.encoding_for_model('gpt-4o')
from openai import OpenAI,AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import setproctitle
setproctitle.setproctitle("gpt-judge")

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

def evaluate_llm_judge(question, gold_answer, generated_answer,use_azure=False):

    if not use_azure:
        model_name="gpt-5-nano"
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY',"your_openai_api_key_here"))
    else:
        model_name = "gpt-5-mini"
        api_key = os.environ.get('AZURE_API_KEY',"your_azure_api_key_here")
        client = AzureOpenAI(api_version="2024-12-01-preview",
                             api_key=api_key,
                             azure_endpoint="",
                             azure_deployment=model_name)

    """Evaluate the generated answer against the gold answer using an LLM judge."""
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
    result=0 if "wrong" in label.lower() else 1

    return result

def process_row(row,is_thinking_model=True, use_azure=False):
    row = row[1]  # Get the row data from the Series
    question = row["question"]
    gold_answer = row["answer"]
    response = row["response"]

    if not isinstance(response, str):
        print("Response is not a string, returning 0")
        return 0

    assert isinstance(question, str) and isinstance(gold_answer, str) and isinstance(response,
                                                                                     str), "Question, gold answer, and generated answers must be strings."

    if is_thinking_model:
        # Keep content after </think>
        if "</think>" not in response:
            print("No </think> tag found in response, returning 0")
            return 0
        final_answer = response.split("</think>")[-1].strip()
        # If final_answer is too long (over 4000 tokens), return 0
        if len(encoding.encode(final_answer)) > 4000:
            print("Short answer (thinking) too long, returning 0")
            return 0

    else:
        final_answer = response
        if len(encoding.encode(final_answer)) > 4000:
            print("Answer (instruct) too long, returning 0")
            return 0

    # Retry if generation fails (up to 3 times, with 1 min delay)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            label = evaluate_llm_judge(question, gold_answer, final_answer, use_azure=use_azure)
            break  # Success, break loop
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(60)  # Wait 1 minute before retry
            else:
                # Max retries reached
                print("Max retries reached. Returning 0.")
                raise Exception("Failed to evaluate response.")

    return label

def eval_df(df, is_thinking_model=True, num_workers=50, use_azure=False, multi_choice=False):

    if "prompt_len" not in df.columns:
        df['prompt_len'] = df['prompt'].apply(lambda x: len(encoding.encode(x)))
    # Explode 'response' column if it's a list (for num_generations > 1)
    num_generations = 1
    if not isinstance(df['response'].iloc[0], str):
        num_generations = len(df['response'].iloc[0])
    df = df.explode('response').reset_index(drop=True)
    print("Expanded df to {} rows due to {} generations per prompt.".format(len(df), num_generations))

    process_row_func=partial(process_row, is_thinking_model=is_thinking_model, use_azure=use_azure)

    if 'judge' not in df.columns:
        if multi_choice:
            # Extract answer from response and compare with gold answer directly
            if is_thinking_model:
                df['short_response'] = df['response'].apply(
                    lambda x: x.split("</think>")[-1].strip() if "</think>" in x else "")
            else:
                df['short_response'] = df['response']
            df['predicted_answer'] = df['short_response'].apply(extract_answer)
            df['judge'] = df.apply(lambda row: row['predicted_answer'].strip().lower() == row['answer'].strip().lower(),
                                   axis=1).astype(int)
        else:
            # Use GPT for evaluation
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_row_func, df.iterrows()), total=len(df), mininterval=5))
            df['judge'] = results


    # 1. Calculate overall accuracy
    accuracy = round(df['judge'].mean(), 3)

    # 2. Calculate accuracy by question_type
    if "question_type" in df.columns:
        accuracy_by_type = df.groupby('question_type')['judge'].mean().to_dict()
        accuracy_by_type = {k: round(v, 3) for k, v in accuracy_by_type.items()}
    else:
        accuracy_by_type = None

    print("Overall accuracy:", accuracy)
    # print("Accuracy by question type:", accuracy_by_type)

    # 3. Calculate accuracy by prompt_length bin
    bins = [0, 16000, 32000, 64000, 128000, float('inf')]
    labels = ['0-16k', '16-32k', '32-64k', '64-128k', '128k+']
    df['prompt_length_bin'] = pd.cut(df['prompt_len'], bins=bins, labels=labels, right=False)
    accuracy_by_prompt_length = df.groupby('prompt_length_bin')['judge'].mean().to_dict()
    accuracy_by_prompt_length = {k: round(v, 3) if not pd.isna(v) else 0 for k, v in accuracy_by_prompt_length.items()}
    # print("Accuracy by prompt length bin:", accuracy_by_prompt_length)

    # 4. Calculate 'think' tag missing ratio
    if is_thinking_model:
        no_think_tag_ratio = round((df['response'].str.contains("</think>") == False).mean(), 3)
    else:
        no_think_tag_ratio = None

    # 5. Calculate response lengths
    df['response_length'] = df['response'].apply(lambda x: len(encoding.encode(x)))
    response_mean_len = round(df['response_length'].mean(), 1)
    response_mean_len_judge_1 = round(df[df['judge'] == 1]['response_length'].mean(), 1) if not df[
        df['judge'] == 1].empty else 0.0

    # 6. Calculate Pass@k (if num_generations > 1)
    accuracy_passk = accuracy  # Default value
    if 'original_index' in df.columns:  # If explode logic was previously executed
        # Re-calculate pass@k
        df_agg = df.groupby('original_index').agg({
            'question': 'first',
            'answer': 'first',
            'response': list,
            'judge': list,
            'question_type': 'first' if 'question_type' in df.columns else (lambda x: None),
            'response_length': list
        }).reset_index(drop=True)

        # Recalculate overall accuracy (Pass@k)
        df_agg['judge_agg'] = df_agg['judge'].apply(lambda x: max(x))  # Correct if at least one is correct
        accuracy_passk = round(df_agg['judge_agg'].mean(), 3)

        # --- Core modification: Build and return the metrics dictionary ---
    metrics = {
        "overall_accuracy": accuracy,
        "accuracy_passk": accuracy_passk,  # Equal to overall_accuracy if no multi-generation
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_prompt_length": accuracy_by_prompt_length,
        "no_think_tag_ratio": no_think_tag_ratio,
        "avg_response_len": response_mean_len,
        "avg_correct_response_len": response_mean_len_judge_1,
        "num_samples": len(df)
    }

    return df, metrics


def extract_answer(response):
    response = response.replace('*', '')
    # Find all matches, take the last one
    match=re.findall(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match[-1]
    else:
        match = re.findall(r'The correct answer is ([A-D])', response)
        if match:
            return match[-1]
        else:
            return ""

if __name__ == '__main__':
    df_list=[
        "./eval_output//longmemeval_qa_Qwen3-4B-Thinking-2507_slide2k_keep10_hf.parquet",
    ]
    for dataset_path in df_list:
        dataset_name=dataset_path.split("/")[-1].replace(".parquet","")
        is_thinking_model="thinking" in dataset_path.lower()
        print("is thinking model:", is_thinking_model)

        print("Evaluating dataset:", dataset_path)
        df = pd.read_parquet(dataset_path)

        # Evaluate
        df,metric=eval_df(df, is_thinking_model=is_thinking_model, num_workers=10, use_azure=True,multi_choice=False)

        # Save results
        df.to_parquet(dataset_path)
        print("Results saved to", dataset_path)