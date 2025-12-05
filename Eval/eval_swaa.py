import os
os.environ['SWAA_DEBUG'] = '0' # Print SWAA debug info (1=yes, 0=no)
os.environ['OPENAI_API_KEY']="" # your openai key
os.environ['AZURE_API_KEY']=""

import sys
import time

# Hack vllm and transformers for SWAA (Sliding Window Attention Acceleration)
import sys
sys.path.append(os.path.dirname((os.path.abspath(__file__))))  # Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path
from swaa_patch import SWAAConfig,hack_hf_swaa,hack_vllm_swaa
hack_hf_swaa(training=False)
hack_vllm_swaa() # you can comment this line if not using vllm

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

from transformers import AutoTokenizer,AutoModelForCausalLM

def split_df_by_length_fair(df, n_splits, col_to_stratify, num_bins=50, random_state=42):
    """
    Randomly split a DataFrame into n_splits, ensuring a fair distribution
    of the column specified by col_to_stratify (e.g., 'prompt length').

    Parameters:
    - df: Original DataFrame
    - n_splits: Number of splits (n)
    - col_to_stratify: Column name used for stratification (e.g., 'prompt_len')
    - num_bins: Number of bins for stratification. Must be at least 2.
    - random_state: Random seed for reproducibility

    Returns:
    - A list of n_splits DataFrames
    """

    # 0. Copy df to avoid modifying original data
    df_copy = df.copy()

    # 1. Create Bins
    # Use pd.qcut to create bins with roughly equal sample sizes
    try:
        df_copy['bin'] = pd.qcut(df_copy[col_to_stratify], q=num_bins, labels=False, duplicates='drop')
    except ValueError as e:
        # qcut may fail if data points are too few or values are too concentrated
        print(f"Failed to create bins (bins={num_bins}): {e}. Attempting with fewer bins.")
        # Try a smaller but still reasonable number of bins
        num_bins = max(2, min(num_bins, len(df_copy[col_to_stratify].unique()) - 1))
        if num_bins < 2:
            print("Cannot create bins. Falling back to pure random split.")
            df_shuffled = df_copy.sample(frac=1, random_state=random_state)
            return np.array_split(df_shuffled.drop(columns=['bin'], errors='ignore'), n_splits)

        df_copy['bin'] = pd.qcut(df_copy[col_to_stratify], q=num_bins, labels=False, duplicates='drop')

    # 2. Global Random Shuffle (!!!Crucial!!!)
    # This ensures samples within the same 'bin' are randomly ordered
    df_shuffled = df_copy.sample(frac=1, random_state=random_state)

    # 3. Fair Assignment
    # Group by 'bin', then number each sample within the group (0, 1, 2, ...)
    # Use the number modulo n_splits to get the fold_id
    df_shuffled['fold_id'] = df_shuffled.groupby('bin').cumcount() % n_splits

    # 4. Split by fold_id
    folds = []
    # Drop the added 'bin' and 'fold_id' columns
    df_to_split = df_shuffled.drop(columns=['bin', 'fold_id'])

    for fold_num in range(n_splits):
        fold_df = df_to_split[df_shuffled['fold_id'] == fold_num]
        folds.append(fold_df)

    # Reset index for each fold
    folds = [fold.reset_index(drop=True) for fold in folds]

    return folds

def vllm_generate(device_id,ds:pd.DataFrame,model_path,max_prompt_len,max_completion_len,num_generations=1,temperature=0.0,vllm_batch_size=32,swaa_config=None):
    print("Running on device:", device_id)
    # Delay start to avoid simultaneous launches
    time.sleep(int(device_id)*3)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    from vllm import LLM, SamplingParams
    from tqdm import tqdm


    # The DataFrame slice is handled by the caller (Pool.starmap)
    ds['response'] = ""  # Initialize response column

    llm = LLM(model=model_path,
              dtype="float16",
              max_model_len=max_prompt_len+max_completion_len,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1,
              trust_remote_code=True,
              enforce_eager=True,
              quantization=None,
              swaa_config=swaa_config,
              #compilation_config=1,

    )

    sampling_params = SamplingParams(
        n=num_generations,temperature=temperature, max_tokens=max_completion_len,top_p=1.0,
    )

    prompts= ds['templated_prompt'].tolist()  # Get all prompts

    # Generate answers in batches
    batch_size=vllm_batch_size
    for i in tqdm(range(0, len(prompts), batch_size), desc="generate answers"):
        batch_prompts = prompts[i:i + batch_size]
        # Generate outputs
        outputs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params, use_tqdm=False)
        llm.reset_prefix_cache()

        if num_generations==1:
            response = [o.outputs[0].text for o in outputs]
            ds.loc[i:i + batch_size - 1, 'response'] = response

            # if i == 0:
            #     print("\nExample Response:", response[0]) # Print first response
        else:
            response = [[o.outputs[j].text for j in range(num_generations)] for o in outputs]
            # # Print first response
            # if i == 0:
            # #     print("\nExample Response:", response[0][0])

            for idx in range(len(response)):
                ds.at[i + idx, 'response'] = response[idx]

    return ds

def hf_generate(device_id,ds:pd.DataFrame,model_path,max_prompt_len,max_completion_len,num_generations=1,temperature=0.0,swaa_config=None,*args,**kwargs):
    print("Running on device:", device_id)
    import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    os.environ['PYTHONOPTIMIZE']= '1'  # Enable optimization mode
    from tqdm import tqdm

    # The DataFrame slice is handled by the caller (Pool.starmap)
    ds['response'] = ""  # Initialize response column

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, add_bos_token=False,
                                              add_eos_token=False)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map={"": device_id},
                                                 dtype="bfloat16",
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 ).eval()

    model.config.swaa_config=swaa_config

    prompts= ds['templated_prompt'].tolist()  # Get all prompts
    for i in tqdm(range(0, len(prompts), 1), desc="generate answers",mininterval=30):
        full_query=prompts[i]
        # Tokenize
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
            # #     print("\nExample Response:", output_text)
            ds.at[i, 'response'] = output_text
        else:
            output_texts=[tokenizer.decode(output_ids[j][len(chat_prompt_ids[0]):], skip_special_tokens=True) for j in range(num_generations)]
            # # Print first response
            # if i == 0 and device_id==0:
            # #     print("\nExample Response:", output_texts[0])
            ds.at[i, 'response'] = output_texts


    return ds

def save_metrics_to_jsonl(file_path, model_name, use_vllm, swaa_config: SWAAConfig, metrics):
    """
    Append evaluation results to a jsonl file.
    Parameters from swaa_config and metrics will be at the root level.
    """
    # 1. Base Info
    record = {
        "model_name": model_name,
        "use_vllm": use_vllm,
    }

    # 2. Flatten swaa_config
    record.update({
        "sliding_window_size": swaa_config.sliding_window_size,
        "keep_first": swaa_config.keep_first,
        "force_fa_decode": swaa_config.force_fa_decode,
        "non_sliding_layers": swaa_config.non_sliding_layers
    })

    # 3. Flatten metrics
    if metrics:
        record.update(metrics)

    # 4. Save
    pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Metrics saved to {file_path}")

    # Remove duplicates from the result file (commented out in original)
    # df=pd.read_json(file_path, lines=True)
    # df['non_sliding_layers'] = df['non_sliding_layers'].apply(tuple)
    # df = df.drop_duplicates(
    #     subset=['model_name', 'sliding_window_size', 'non_sliding_layers', 'keep_first', 'force_fa_decode'],
    #     keep='last')
    # df.to_json(file_path, orient='records', lines=True, force_ascii=False)


def main(swaa_config:SWAAConfig,model_list,dataset_path):
    mark=swaa_config.mark
    generate_func = hf_generate if not use_vllm else vllm_generate
    mark=mark+ ("_hf" if not use_vllm else "_vllm")


    # Content appended to each prompt (e.g., "Please answer briefly")
    addition_prompt =("\n\nTips: Since the context is very long, you should think step by step and answer carefully based on the context. "
                       "You'd better carefully review the whole context to find relevant information for many times to avoid omission.") #"""#"  #
    addition_prompt =""
    mark = mark+"_CoTprompt" if addition_prompt != "" else mark

    if num_generations>1:
        mark=mark+"_ngen"+str(num_generations)

    dataset_name = dataset_path.split("/")[-1].replace(".parquet", "")

    # Evaluation result save path
    result_jsonl_path="./eval_output/result_{dataset_name}.jsonl".format(dataset_name=dataset_name)

    for model_path in model_list:
        device_list=device_list_all.copy()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_name=model_path.split("/")[-1] if ("checkpoint" not in model_path and "merged" not in model_path) else "-".join(model_path.split("/")[-2:])
        is_thinking_model="thinking" in model_name.lower()
        if is_thinking_model:
            max_completion_len = 10000
        else:
            max_completion_len = 2000

        output_path = "./eval_output/{dataset_name}/{model_name}_{mark}.parquet".format(
            dataset_name=dataset_name,
            model_name=model_name,
            mark=mark)
        # Skip if output_path already exists and force_recompute is False
        if not force_recompute and os.path.exists(output_path):
            print("Output already exists, directly read:", output_path)
            ds = pd.read_parquet(output_path)
            print("number of samples in existing output:", len(ds))

        else:
            print("model:", model_path)
            print("dataset:", dataset_path)
            print("output_path:", output_path)

            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            ds = pd.read_parquet(dataset_path)

            if 'locomo' in dataset_name.lower():
                # Remove rows where question_type is 5
                ds = ds[ds['question_type'] != 5]
                ds.reset_index(drop=True, inplace=True)  # Reset index
                print("After removing question_type 5, number of samples:", len(ds))

            print("number of samples:", len(ds))

            if "prompt" not in ds.columns and "context" in ds.columns:
                # Concatenate context and question into a new prompt column: "Context: {context}\nQuestion: {question}"
                ds['prompt'] = "Context: " + ds['context'] + "\n\nQuestion: " + ds['question']

            # Append addition_prompt to the prompt
            ds["templated_prompt"]=ds['prompt'].apply(lambda x: tokenizer.apply_chat_template([{"role":"user","content":x+addition_prompt}],tokenize=False,add_generation_prompt=True,))

            # Calculate prompt length and filter out prompts longer than max_prompt_len
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
                # Split df into process_num folds
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
            # Merge the 'response' column back to ds using 'index' for alignment
            if 'response' in ds.columns:
                ds = ds.drop(columns=['response'])
            ds = ds.merge(result_df[['index', 'response']], on='index', how='left')

            # Save storage space by dropping 'index', 'prompt', 'context', etc. (if they exist)
            ds = ds.drop(columns=['index'], errors='ignore')
            ds = ds.drop(columns=['prompt'], errors='ignore')
            ds = ds.drop(columns=['context'], errors='ignore')
            ds = ds.drop(columns=['templated_prompt'], errors='ignore')

            # Temporarily save
            ds.to_parquet(output_path)
            print("Output saved to:", output_path)

            if get_accuracy:
                try:
                    from gpt_eval import eval_df

                    # --- Modification: eval_df now returns (ds, metrics) ---
                    ds, metrics = eval_df(ds,
                                          is_thinking_model=is_thinking_model,
                                          use_azure=use_azure,
                                          num_workers=10,
                                          multi_choice=True if 'longbench' in dataset_name.lower() else False,
                                          )

                    print("Output with accuracy saved to:", output_path)
                    ds.to_parquet(output_path, index=False)  # Save parquet with judge details

                    # --- Modification: Call new save function ---
                    save_metrics_to_jsonl(
                        file_path=result_jsonl_path,
                        model_name=model_name,
                        use_vllm=use_vllm,
                        swaa_config=swaa_config,
                        metrics=metrics,  # Pass metrics for flattening
                    )

                except Exception as e:
                    print("Error in accuracy evaluation:", e)
                    import traceback
                    traceback.print_exc()

def get_settings_from_json(json_path)->list:
    with open(json_path, 'r', encoding='utf-8') as f:
        settings_list = json.load(f)
    swaa_settings = []
    for setting in settings_list:
        swaa_config = SWAAConfig(**setting['swaa_config'])
        swaa_settings.append({
            'swaa_config': swaa_config,
            'model_list': setting.get('model_list', []),
        })
    return swaa_settings

if __name__ == '__main__':
    device_list_all = [0,1,2,3,4,5,6,7]  #[1]# [0]#  # GPUs to use
    use_vllm=True # Use vllm for generation, otherwise use Hugging Face transformers
    force_recompute=False  # Force recomputing existing output files


    dataset_path = "../Datasets/longmemeval_24k.parquet" # Select a dataset to test
    #dataset_path = "../Datasets/longbenchv2_qa.parquet"

    sample_first=500  # If not None, only use the first sample_first data points for testing
    max_prompt_len = 128000 # 40000#
    num_generations = 1  # Number of answers to generate per prompt


    vllm_batch_size = 64
    temperature = 1.0 if num_generations > 1 else 0.0

    get_accuracy = True  # Evaluate answer correctness using GPT
    use_azure = True

    model_list=[
             "/share/models/Qwen3-4B-Thinking-2507",
             "/share/models/Qwen3-4B-Instruct-2507",
             "/share/models/Meta-Llama-3.1-8B-Instruct",
             "/share/models/Qwen3-30B-A3B-Thinking-2507",
             "/share/models/Qwen3-30B-A3B-Instruct-2507",
         ]

    # Read settings_list from json file
    settings_list = get_settings_from_json("settings_list/non_sft_settings_4b.json")
    #settings_list=settings_list+get_settings_from_json("non_sft_settings_30b.json")
    print("Starting evaluation of all settings...")
    print(settings_list)

    for settings in settings_list:
        main(settings["swaa_config"], settings['model_list'], dataset_path)


# nohup python -u eval_swaa.py > vllm_eval_sft.log 2>&1 &