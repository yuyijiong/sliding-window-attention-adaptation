import os
import sys
import time
import json

sys.path.append("../Eval")
from eval_swaa import main,SWAAConfig

os.environ['SWAA_DEBUG'] = '0'
os.environ['OPENAI_API_KEY'] = ""
os.environ['AZURE_API_KEY'] = ""
gpu_memory_utilization = 0.9


if __name__ == '__main__':
    device_list_all = [0,1,2,3]  # GPUs to use
    use_vllm=True
    force_recompute=False  # Whether to force re-computation of existing output files

    dataset_path = "../Datasets/longmemeval_24k.parquet" # Select a dataset to test
    #dataset_path = "../Datasets/longbenchv2_qa.parquet"

    sample_first=500  # If not None, only use the first 'sample_first' samples
    max_prompt_len = 128000 #40000#
    num_generations = 1  # Number of responses to generate per prompt


    vllm_batch_size = 64
    temperature = 1.0 if num_generations > 1 else 0.0

    get_accuracy = True  # Whether to use GPT to evaluate answer correctness
    use_azure = True

    model_list=[
             "/share/models/Qwen3-4B-Thinking-2507",
             "/share/models/Qwen3-4B-Instruct-2507",
             "/share/models/Meta-Llama-3.1-8B-Instruct",
             "/share/models/Qwen3-30B-A3B-Thinking-2507",
             "/share/models/Qwen3-30B-A3B-Instruct-2507",
         ]
    settings_list=[]

    for model in model_list:
        model_name=model.split("/")[-1] if "checkpoint" not in model else "-".join(model.split("/")[-2:])
        # Read selected_lazy_layers and non_lazy_layers from model__lazy_layers_freq.json
        with open(f"{model_name}_lazy_layers_freq.json", 'r', encoding='utf-8') as f:
            lazy_info = json.load(f)
        selected_lazy_layers=lazy_info.get("selected_lazy_layers",[])
        non_lazy_layers=lazy_info.get("non_lazy_layers",[])
        for non_sliding_layers in [list(range(0,50,2)),list(range(1,50,2)),non_lazy_layers,selected_lazy_layers]:
            settings_list.append({
                'swaa_config': SWAAConfig(
                    sliding_window_size=2000,
                    keep_first=100,
                    force_fa_decode=True,
                    non_sliding_layers=non_sliding_layers,
                ),
                'model_list': [model],
            })

    print("Starting evaluation of all settings...")
    print(settings_list)

    for settings in settings_list:
        main(settings["swaa_config"], settings['model_list'], dataset_path)

