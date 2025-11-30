import os
import sys
import time

import multiprocessing
from multiprocessing import Pool
import json
import pandas as pd
import numpy as np
import tiktoken
import pathlib
import setproctitle
setproctitle.setproctitle("swaa_eval")
encoding = tiktoken.encoding_for_model('gpt-4o')
multiprocessing.set_start_method("spawn", force=True)
from tqdm import tqdm

from transformers import AutoTokenizer,AutoModelForCausalLM

def split_df_by_length_fair(df, n_splits, col_to_stratify, num_bins=50, random_state=42):
    """
    Splits a DataFrame into n_splits parts (without using sklearn),
    ensuring a fair distribution of the 'col_to_stratify' column's mean
    across all resulting splits.

    Args:
    - df: Original DataFrame
    - n_splits: Number of splits (n)
    - col_to_stratify: Column name for stratification (e.g., 'prompt length')
    - num_bins: Number of bins for stratification. Should be >= n_splits.
    - random_state: Random seed for reproducibility

    Returns:
    - A list containing n_splits DataFrames
    """

    # 0. Copy df to avoid modifying original data
    df_copy = df.copy()

    # 1. Create Bins (Binning)
    # Use pd.qcut to create bins with roughly equal sample size
    try:
        df_copy['bin'] = pd.qcut(df_copy[col_to_stratify], q=num_bins, labels=False, duplicates='drop')
    except ValueError as e:
        # qcut might fail if data points are too few or values are too concentrated
        print(f"Binning failed (bins={num_bins}): {e}. Trying with fewer bins.")
        # Try a smaller but still reasonable number of bins
        num_bins = max(2, min(num_bins, len(df_copy[col_to_stratify].unique()) - 1))
        if num_bins < 2:
            print("Could not create bins. Falling back to purely random split.")
            df_shuffled = df_copy.sample(frac=1, random_state=random_state)
            return np.array_split(df_shuffled.drop(columns=['bin'], errors='ignore'), n_splits)

        df_copy['bin'] = pd.qcut(df_copy[col_to_stratify], q=num_bins, labels=False, duplicates='drop')

    # 2. Global Shuffling (!!!Crucial!!!)
    # Ensures samples within the same 'bin' are randomly ordered
    df_shuffled = df_copy.sample(frac=1, random_state=random_state)

    # 3. Fair Assignment
    # Group by 'bin', then number each sample (0, 1, 2, ...) within the group
    # Take modulo n_splits to get the fold_id
    df_shuffled['fold_id'] = df_shuffled.groupby('bin').cumcount() % n_splits

    # 4. Split based on fold_id
    folds = []
    # Drop the added 'bin' and 'fold_id' columns
    df_to_split = df_shuffled.drop(columns=['bin', 'fold_id'])

    for fold_num in range(n_splits):
        fold_df = df_to_split[df_shuffled['fold_id'] == fold_num]
        folds.append(fold_df)

    # Reset index for each split
    folds = [fold.reset_index(drop=True) for fold in folds]

    return folds

def vllm_generate(device_id,ds:pd.DataFrame,model_path,max_prompt_len,max_completion_len,num_generations=1,temperature=0.0,vllm_batch_size=64):
    print("Running on device:", device_id)
    # Delay to avoid simultaneous startup
    time.sleep(int(device_id)*3)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    from vllm import LLM, SamplingParams

    # The DataFrame slice is handled outside of this function
    # ds = df[process_id::process_num].copy().reset_index(drop=True)
    ds['response'] = ""  # Initialize the response column

    llm = LLM(model=model_path,
              dtype="float16",
              max_model_len=max_prompt_len+max_completion_len,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              trust_remote_code=True,
              enforce_eager=True,
              quantization=None,

    )

    sampling_params = SamplingParams(
        n=num_generations,temperature=temperature, max_tokens=max_completion_len,top_p=1.0,
    )

    prompts= ds['templated_prompt'].tolist()  # Get all prompts

    # Generate answers in batches
    batch_size=vllm_batch_size
    for i in tqdm(range(0, len(prompts), batch_size), desc="generate answers"):
        batch_prompts = prompts[i:i + batch_size]
        # Generate output
        outputs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params, use_tqdm=False)
        llm.reset_prefix_cache()

        if num_generations==1:
            response = [o.outputs[0].text for o in outputs]
            ds.loc[i:i + batch_size - 1, 'response'] = response

            # if i == 0:
            #     print("\nExample Response:", response[0])            # Print first response
        else:
            response = [[o.outputs[j].text for j in range(num_generations)] for o in outputs]
            # # Print first response
            # if i == 0:
            #     print("\nExample Response:", response[0][0])            # Print first response

            for idx in range(len(response)):
                ds.at[i + idx, 'response'] = response[idx]

    return ds

def hf_generate(device_id,ds:pd.DataFrame,model_path,max_prompt_len,max_completion_len,num_generations=1,temperature=0.0,*args,**kwargs):
    print("Running on device:", device_id)
    # The DataFrame slice is handled outside of this function
    ds['response'] = ""  # Initialize the response column

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_bos_token=False,
                                              add_eos_token=False)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map={"": device_id},
                                                 dtype="bfloat16",
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 ).eval()

    prompts= ds['templated_prompt'].tolist()  # Get all prompts
    for i in tqdm(range(0, len(prompts), 1), desc="generate answers",mininterval=30):
        full_query=prompts[i]
        # tokenize
        chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": full_query}], tokenize=False,
                                                    add_generation_prompt=True)
        chat_prompt_ids = tokenizer(chat_prompt, return_tensors="pt")["input_ids"].to(model.device)

        output_ids = model.generate(input_ids=chat_prompt_ids[:, :], max_new_tokens=max_completion_len,
                                    do_sample=True if temperature > 0 else False,
                                    temperature=temperature,
                                    use_cache=True,
                                    top_p=1.0,
                                    num_return_sequences=num_generations,
                                    return_dict_in_generate=False, )

        if num_generations==1:
            output_text=tokenizer.decode(output_ids[0][len(chat_prompt_ids[0]):], skip_special_tokens=True)
            # # Print first response
            # if i == 0 and device_id==0:
            #     print("\nExample Response:", output_text)
            ds.at[i, 'response'] = output_text
        else:
            output_texts=[tokenizer.decode(output_ids[j][len(chat_prompt_ids[0]):], skip_special_tokens=True) for j in range(num_generations)]
            # # Print first response
            # if i == 0 and device_id==0:
            #     print("\nExample Response:", output_texts[0])
            ds.at[i, 'response'] = output_texts


    return ds

def main(model_list,dataset_path):
    generate_func = hf_generate if not use_vllm else vllm_generate
    mark=("_hf" if not use_vllm else "_vllm")

    # Additional prompt to append to each prompt (e.g., "Please answer briefly")
    addition_prompt =("\n\nTips: Since the context is very long, you should think step by step and answer carefully based on the context. "
                       "You'd better carefully review the whole context to find relevant information for many times to avoid omission.") #"""#"  #
    addition_prompt =""
    mark = mark+"_CoTprompt" if addition_prompt != "" else mark

    if num_generations>1:
        mark=mark+"_ngen"+str(num_generations)

    dataset_name = dataset_path.split("/")[-1].replace(".parquet", "")

    for model_path in model_list:
        device_list=device_list_all.copy()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_name=model_path.split("/")[-1] if "checkpoint" not in model_path else "-".join(model_path.split("/")[-2:])
        is_thinking_model="thinking" in model_name.lower()
        if is_thinking_model:
            max_completion_len = 10000
        else:
            max_completion_len = 4000

        output_path = "./self_distilled_data/{dataset_name}/{model_name}_{mark}.parquet".format(
            dataset_name=dataset_name,
            model_name=model_name,
            mark=mark)

        # Skip if output_path already exists and force_recompute is False
        if not force_recompute and os.path.exists(output_path):
            print("Output already exists, skipping:", output_path)
            continue

        print("model:", model_path)
        print("dataset:", dataset_path)
        print("output_path:", output_path)

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        ds = pd.read_parquet(dataset_path)

        print("number of samples:", len(ds))

        if "prompt" not in ds.columns and "context" in ds.columns:
            # Concatenate context and question into a new prompt column, format: "Context: {context}\nQuestion: {question}"
            ds['prompt'] = "Context: " + ds['context'] + "\n\nQuestion: " + ds['question']

        # Append addition_prompt to the prompt
        ds["templated_prompt"]=ds['prompt'].apply(lambda x: tokenizer.apply_chat_template([{"role":"user","content":x+addition_prompt}],tokenize=False,add_generation_prompt=True,))

        # Calculate prompt length and filter out samples exceeding max_prompt_len
        if "prompt_len" not in ds.columns and "prompt_length" not in ds.columns:
            ds['prompt_len'] = ds['templated_prompt'].swifter.apply(lambda x: len(tokenizer.encode(x)))
        elif "prompt_len" not in ds.columns and "prompt_length" in ds.columns:
            ds = ds.rename(columns={"prompt_length": "prompt_len"})
        ds = ds[ds['prompt_len'] <= max_prompt_len]
        ds.reset_index(drop=True, inplace=True)  # Reset index
        print("number of samples after filtering:", len(ds))

        if sample_first is not None:
            ds = ds[0:sample_first]
            print("number of samples after sampling:", len(ds))

        # Add an index column
        ds['index'] = ds.index

        process_num=len(device_list)
        print("Starting evaluation for",output_path)
        if process_num>1:
            # Split the DataFrame into process_num parts
            num_bins = 5
            ds_folds = split_df_by_length_fair(ds,
                                               n_splits=process_num,
                                               col_to_stratify='prompt_len',
                                               num_bins=num_bins,
                                               random_state=0)

            with Pool(processes=process_num) as pool:
                results = pool.starmap(generate_func,
                                       [(device_id, sub_ds, model_path, max_prompt_len, max_completion_len, num_generations, temperature,vllm_batch_size,swaa_config)
                                        for device_id, sub_ds in zip(device_list, ds_folds)])
        else:
            results = [generate_func(device_list[0], ds, model_path, max_prompt_len, max_completion_len, num_generations, temperature,vllm_batch_size,swaa_config)]

        # Merge results
        result_df = pd.concat(results, ignore_index=True)
        # Assign 'response' column from result_df to ds based on 'index'
        if 'response' in ds.columns:
            ds = ds.drop(columns=['response'])
        ds = ds.merge(result_df[['index', 'response']], on='index', how='left')

        # Save temporary results
        ds.to_parquet(output_path)
        print("output saved to:", output_path)

        try:
            from vllm_judge import vllm_eval

            # Use local judge model for accuracy evaluation
            print("Starting accuracy evaluation with judge model...")
            ds = vllm_eval(ds,
                            judge_model_path="/share/models/Qwen3-30B-A3B-Instruct-2507",
                            device_list=device_list,
                                  )

            print("output with accuracy saved to:", output_path)
            ds.to_parquet(output_path, index=False)  # Save parquet with judge details

        except Exception as e:
            print("Error in accuracy evaluation:", e)
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    device_list_all = [0,1,2,3,4,5,6,7]   # Which GPUs to use
    use_vllm=True
    force_recompute=False  # Force recompute if output file already exists

    dataset_path = "../Datasets/fusang_long.parquet" # Select the first dataset for testing

    sample_first=500  # If not None, only use the first sample_first data points for testing
    max_prompt_len = 40000 #128000 #
    num_generations = 1  # Number of answers to generate per prompt


    vllm_batch_size = 32
    temperature = 1.0 if num_generations > 1 else 0.0

    model_list=[
             "/share/models/Qwen3-4B-Thinking-2507",
             "/share/models/Qwen3-4B-Instruct-2507",
             "/share/models/Meta-Llama-3.1-8B-Instruct",
             "/share/models/Qwen3-30B-A3B-Thinking-2507",
             "/share/models/Qwen3-30B-A3B-Instruct-2507",
         ]

    main(model_list=model_list,
         dataset_path=dataset_path)