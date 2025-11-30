import os
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from transformers import AutoTokenizer

import sys
sys.path.append("../")

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

import pandas as pd
import tiktoken
encoding = tiktoken.encoding_for_model('gpt-4o')
from tqdm import tqdm
import numpy as np

import setproctitle
setproctitle.setproctitle("vllm_judge")

def vllm_judge(device_id,ds:pd.DataFrame,model_path,max_prompt_len=10000,max_completion_len=1000,num_generations=1,temperature=0.0,vllm_batch_size=32):
    print("Running on device:", device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    from vllm import LLM, SamplingParams

    # The DataFrame slice is handled outside of this function
    ds['response'] = ""  # Initialize the response column

    llm = LLM(model=model_path,
              dtype="float16",
              max_model_len=max_prompt_len+max_completion_len,
              gpu_memory_utilization=0.5,
              tensor_parallel_size=1,
              trust_remote_code=True,
              enforce_eager=True,
              quantization=None
              )

    sampling_params = SamplingParams(
        n=num_generations,temperature=temperature, max_tokens=max_completion_len,top_p=1.0,
    )

    prompts= ds['prompt_for_judge'].tolist()  # Get all prompts

    # Generate answers in batches
    batch_size=vllm_batch_size
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        # Generate output
        outputs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params, use_tqdm=False)
        if num_generations==1:
            response = [o.outputs[0].text for o in outputs]
            # if i == 0:
            #     print("\nExample Response:", response[0])            # Print first response
        else:
            response = [[o.outputs[j].text for j in range(num_generations)] for o in outputs]
            # Print first response
            # if i == 0:
            #     print("\nExample Response:", response[0][0])            # Print first response
        ds.loc[i:i + batch_size-1, 'response'] = response

    return ds

def vllm_eval(ds,judge_model_path,device_list)->pd.DataFrame:

    # If response is a list, explode it into multiple rows, copying other columns
    if not isinstance(ds['response'].iloc[0], str):
        ds = ds.explode('response').reset_index(drop=True)

    ds.reset_index(drop=True, inplace=True)  # Reset index
    print("number of samples:", len(ds))

    if not is_thinking_model:
        ds['prompt_for_judge'] = ds.swifter.apply(lambda row: ACCURACY_PROMPT.format(
            question=row['question'],
            gold_answer=row['answer'],
            generated_answer=row['response']
        ), axis=1)
    else:
        # If it's a thinking model, check if the response contains </think> tag
        def process_thinking_response(row):
            response = row['response']
            if "</think>" in response:
                # Extract content after </think> as the final answer
                final_answer = response.split("</think>")[-1].strip()
            else:
                final_answer = "None"  # Assume no answer if </think> is missing
            prompt = ACCURACY_PROMPT.format(
                question=row['question'],
                gold_answer=row['answer'],
                generated_answer=final_answer
            )
            return prompt

        ds['prompt_for_judge'] = ds.swifter.apply(process_thinking_response, axis=1)

    # apply template to prompt_for_judge
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
    ds['prompt_for_judge'] = ds['prompt_for_judge'].swifter.apply(
        lambda x: tokenizer.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True, ))

    # Filter out samples where prompt_for_judge length exceeds 10000
    ds['prompt_len'] = ds['prompt_for_judge'].swifter.apply(lambda x: len(tokenizer.encode(x)))
    ds = ds[ds['prompt_len'] < 10000]
    ds.reset_index(drop=True, inplace=True)  # Reset index
    print("number of samples after filtering:", len(ds))

    # Add an index column
    ds['index'] = ds.index
    # Multi-GPU generation
    devices_num=len(device_list)

    from long_mem_eval.gpu_memory_check import check_gpu_memory_usage
    gpus_are_ready = check_gpu_memory_usage(
        threshold_percentage=5,
        check_interval_seconds=60 * 20,
        gpus=device_list
    )

    process_num = len(device_list)

    if devices_num>1:
        # Split the DataFrame into process_num parts
        ds_folds = np.array_split(ds, process_num)
        from multiprocessing import Pool
        with Pool(processes=devices_num) as pool:
            results = pool.starmap(vllm_judge,
                                   [(device_id, sub_ds, judge_model_path,) for device_id, sub_ds in zip(device_list, ds_folds)])
    else:
        results = [vllm_judge(device_list[0], ds, judge_model_path,)]

    # Merge results
    result_df = pd.concat(results, ignore_index=True)
    # Assign 'response' column from result_df to ds based on 'index'
    if 'response' in ds.columns:
        ds = ds.drop(columns=['response'])
    ds = ds.merge(result_df[['index', 'response']], on='index', how='left')

    # Convert CORRECT/WRONG string in response column to 1/0 in judge column
    ds['judge'] = ds['response'].apply(lambda x:0 if "WRONG" in x.upper() else 1)

    # Drop redundant columns
    ds = ds.drop(columns=['index', 'prompt_for_judge','templated_prompt'])

    return ds

if __name__ == '__main__':

    model_path="/share/models/Qwen3-4B-Instruct-2507"
    model_name=model_path.split("/")[-1] if "checkpoint" not in model_path else "-".join(model_path.split("/")[-2:])

    dataset_path = "//share/yyj/llm_as_memory/long_mem_eval/output/fusang_long_Qwen3-30B-A3B-Thinking-2507__vllm_ngen4.parquet"
    is_thinking_model="thinking" in dataset_path.lower()
    print("is thinking model:", is_thinking_model)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Evaluating dataset:", dataset_path)
    ds = pd.read_parquet(dataset_path)
    ds = vllm_eval(ds, judge_model_path=model_path,device_list=[0,1,2,3,4,5,6,7])


    # Save results
    ds.to_parquet(dataset_path.replace(".parquet", f"_{model_name}_judged.parquet"), index=False)
    print("output saved to:", dataset_path.replace(".parquet", f"_{model_name}_judged.parquet"))

    # Calculate overall accuracy
    accuracy = ds['judge'].mean()
    print("Overall accuracy:", accuracy)