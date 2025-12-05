import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"]="SWAA_SFT"
os.environ["SWAA_DEBUG"]="0"  #1 or 0

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory to path
# hack vllm and transformers for SWAA
from swaa_patch import SWAAConfig,hack_hf_swaa
hack_hf_swaa(training=True)

import pathlib
import torch
from transformers import LlamaTokenizer, TrainingArguments,Trainer,AutoTokenizer, Qwen3ForCausalLM,AutoConfig
from datasets import Dataset,concatenate_datasets
from transformers import DataCollatorForSeq2Seq,BitsAndBytesConfig,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import setproctitle
setproctitle.setproctitle("sft-swaa")

def encode_hf(example, tokenizer:LlamaTokenizer,max_length=4096,min_length=1000):

    # Handle 'conversations' field
    if 'conversations' in example:
        conversations = example['conversations']

    elif "prompt" in example and "response" in example:
        conversations = [{"role": "user", "content": example["prompt"]},
                         {"role": "assistant", "content": example["response"]}]

    elif "instruction" in example and "output" in example:
        conversations = [{"role": "user", "content": example["instruction"]},
                         {"role": "assistant", "content": example["output"]}]
    else:
        raise ValueError("Example must contain 'conversations' or 'instruction'/'output' fields.")

    # Convert conversations to token IDs
    ids = tokenizer.apply_chat_template(conversations, add_generation_prompt=False, tokenize=True,
                                        max_length=max_length, return_assistant_tokens_mask=True,return_dict=True,truncation=False,)

    # Skip examples that are too long or too short
    if len(ids['input_ids']) > max_length or len(ids['input_ids']) < min_length:
        print(f"Example too long or too short: {len(ids['input_ids'])} > {max_length}, skipping.")
        return {"input_ids": None, "attention_mask": None, "labels": None}

    # Generate labels
    labels = ids['input_ids'].copy()
    # Replace tokens outside of the assistant's turn with -100 using 'assistant_masks'
    for i, mask in enumerate(ids['assistant_masks']):
        if mask == 0:
            labels[i] = -100

    return {"input_ids": ids['input_ids'], "attention_mask": ids['attention_mask'], "labels": labels}


def main(model_path,dataset_list,swaa_config:SWAAConfig):
    # Print Process ID
    print("Process ID:", os.getpid())

    model_name=model_path.split("/")[-1] if "checkpoint" not in model_path else "-".join(model_path.split("/")[-2:])
    output_dir="./training_output/{}-sft-fusang-{}".format(model_name,swaa_config.mark)

    train_args = TrainingArguments(
        report_to="wandb",#"none",#
        output_dir=output_dir,
        run_name=output_dir,  # name of the W&B run (optional)
        optim="adamw_torch",
        learning_rate=1e-4,
        lr_scheduler_type='linear',
        warmup_ratio=0.05,
        num_train_epochs=1,
        eval_steps=500,
        #save_steps=0.2,  # Save every 0.2 steps
        eval_strategy='no',
        save_strategy='epoch',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        bf16=True,
        fp16=False,
        deepspeed=None,
        #deepspeed="ds_config.json",
        seed=0,
        save_only_model=True,
        save_total_limit=11,
        #torch_compile=True,
        #fsdp="full_shard",#"shard_grad_op",#
        #fsdp_config=fsdp_config,
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,

        group_by_length=True,
        length_column_name="length",

    )

    # Model and Tokenizer setup
    max_length=34000 if "4b" in model_name.lower() else 28000
    min_length=10

    tokenizer_path=model_path
    config= AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer_train = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Set chat_template
    if "thinking" in model_path.lower():
        print("Using thinking template")
        with open("./qwen3-thinking-template", "r", encoding="utf-8") as f:
            template = f.read()
    else:
        print("Using instruct template")
        with open("./qwen3-instruct-template", "r", encoding="utf-8") as f:
            template = f.read()

    tokenizer_train.chat_template = template

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],#'all-linear',#
        lora_dropout=0.1,
        bias="none",
        #init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    # Load dataset, concatenate, and filter for judge==1
    ds_ori=concatenate_datasets([Dataset.from_parquet(ds,columns=['question','answer','prompt','response','judge']) for ds in dataset_list])#.select(range(100))

    # Keep only samples where judge == 1
    ds_ori=ds_ori.filter(lambda x: x['judge'] == 1, num_proc=20, desc="Filtering judge==1 examples")
    print("Initial dataset size:",len(ds_ori))

    with train_args.main_process_first(desc="Loading dataset"):
        ds=ds_ori.map(lambda e: encode_hf(e,tokenizer_train,max_length=max_length,min_length=min_length),
                      remove_columns=ds_ori.column_names,
                      num_proc=16,
                      desc="Encoding dataset",
                      load_from_cache_file=True)

        # Remove examples where input_ids is None
        ds=ds.filter(lambda x: x['input_ids'] is not None, num_proc=20, desc="Filtering invalid examples")
        print("Final dataset size:",len(ds))
        # Calculate length
        ds=ds.map(lambda x: {'length': len(x['input_ids'])},
                    num_proc=32,
                    desc="Calculating length")

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                             dtype=torch.bfloat16,
                                             trust_remote_code=True,
                                             use_cache=False,
                                            #quantization_config=bnb_config,
                                             attn_implementation="flash_attention_2",#"sdpa",#
                                             ).eval()

    model.config.swaa_config=swaa_config

    model.gradient_checkpointing_enable()
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(
                       tokenizer=tokenizer,
        ),
        train_dataset=ds,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=False)
    # Merge LoRA weights and save in main process
    if train_args.distributed_state.is_main_process and not trainer.is_fsdp_enabled:
        merged_model_path=output_dir+"/final-merged"
        pathlib.Path(merged_model_path).mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        # 6. Save the merged model
        print(f"Saving merged model to: {merged_model_path}...")
        # swaa_config does not need to be saved
        merged_model.config.swaa_config = None
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)


if __name__ == '__main__':
    import time
    sft_model_path="/share/models/Qwen3-4B-Instruct-2507"#"/share/models/Qwen3-4B-Thinking-2507"#"/share/models/Qwen3-0.6B"#"/data/models/Qwen3-0.6B"#  #

    sft_ds_list_all=[
    "//share/yyj/llm_as_memory/long_mem_eval/output/fusang_long_Qwen3-4B-Thinking-2507__vllm_ngen4_addprompt_Qwen3-4B-Instruct-2507__judged_dedup_shortest3.parquet",
    "//share/yyj/llm_as_memory/long_mem_eval/output/fusang_long_Qwen3-4B-Instruct-2507__vllm_ngen4_Qwen3-4B-Instruct-2507__judged.parquet",
    "//share/yyj/llm_as_memory/long_mem_eval/output/fusang_long_Qwen3-30B-A3B-Thinking-2507__vllm_ngen4_Qwen3-4B-Instruct-2507__judged.parquet",
    "//share/yyj/llm_as_memory/long_mem_eval/output/fusang_long_Qwen3-30B-A3B-Instruct-2507__vllm_ngen4_Qwen3-4B-Instruct-2507__judged.parquet",
    ]

    # Select the corresponding dataset
    if "qwen3-4b-thinking" in sft_model_path.lower():
        sft_ds_list=sft_ds_list_all[0:1]
    elif "qwen3-4b-instruct" in sft_model_path.lower():
        sft_ds_list=sft_ds_list_all[1:2]
    elif "qwen3-30b-a3b-thinking" in sft_model_path.lower():
        sft_ds_list=sft_ds_list_all[2:3]
    elif "qwen3-30b-a3b-instruct" in sft_model_path.lower():
        sft_ds_list=sft_ds_list_all[3:4]
    else:
        raise NotImplementedError("Unknown model for selecting dataset")

    # Get available GPU list from CUDA_VISIBLE_DEVICES environment variable
    device_list=os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device_list:
        device_list = [int(x) for x in device_list.split(",")]
    else:
        raise "CUDA_VISIBLE_DEVICES environment variable is not set, cannot get GPU list"

    sft_swaa_config=SWAAConfig(
        sliding_window_size=2048,
        keep_first=0,
        force_fa_decode=True,
        non_sliding_layers=list(range(1,50,4))#[]#
    )

    main(model_path=sft_model_path,
         dataset_list=sft_ds_list,
         swaa_config=sft_swaa_config)

# launch command examples:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup accelerate launch --config_file ddp_config.yaml -num_processes 8 sft_swaa.py >sliding-sft.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup accelerate launch --config_file fsdp_config.yaml --num_processes 8 sft_swaa.py >sliding-sft.log 2>&1 &