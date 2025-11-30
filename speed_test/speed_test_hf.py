import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("../Patch")
import time

import torch
import numpy as np
import multiprocessing
import json
import pathlib
multiprocessing.set_start_method('spawn', force=True)
torch.manual_seed(0)
np.random.seed(0)

from hack_hf_swaa import hack_hf_swaa,SWAAConfig
hack_hf_swaa(training=False)

from transformers import AutoTokenizer, AutoConfig,AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

def hf_model_generate(model,tokenizer,prompts:list[str],generation_config):

    inputs = tokenizer(prompts)
    input_ids_list = inputs["input_ids"]
    start_time = time.time()
    with torch.no_grad():
        if "paged" in model.config._attn_implementation:
            # Code block for 'paged_attention' (removed as it's commented out in config)
            # outputs = model.generate_batch(inputs=input_ids_list,
            #                                generation_config=generation_config
            #                          )
            # output_text = [tokenizer.decode(output.generated_tokens) for output in outputs.values()]
            pass # Currently unused branch

        else:
            output_text = []
            for input_ids in tqdm(input_ids_list, desc="Generating one by one with HF"):
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
                outputs = model.generate(input_ids=input_ids,
                                        generation_config=generation_config
                                     )

                decoded = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                output_text.append(decoded)

    print("Average HF generation time:", round((time.time() - start_time)/len(prompts),4))

    return output_text


def main(model_path, swaa_config:SWAAConfig):

    tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 attn_implementation="flash_attention_2",  #"paged_attention",#
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 device_map="cuda")
    model.config.swaa_config = swaa_config

    generation_config=GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_return_sequences=1,
        max_new_tokens=2000,
        min_new_tokens=2000,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        return_dict_in_generate=False,
        output_logits=False,
    )

    prompt_length=64000
    sample_num=3
    random_prompt_list=[tokenizer.decode(np.random.randint(10, 2000, size=(prompt_length))) for _ in range(sample_num)]

    print("Testing flash_attention... SWAA config:", swaa_config)
    response = hf_model_generate(model, tokenizer, prompts=random_prompt_list,generation_config=generation_config)


if __name__ == '__main__':

    # Replaced sensitive path with a placeholder (based on context of model name)
    model_path = "/share/models/Qwen3-0.6B"

    settings_list = [
        {"sliding_window_size": None, "non_sliding_layers": []},
        {"sliding_window_size": 2000, "non_sliding_layers": []},
        {"sliding_window_size": 2000, "non_sliding_layers": [],'force_fa_decode':True},
        {"sliding_window_size": 2000, "non_sliding_layers": [], 'force_fa_decode': list(range(0, 50, 2))},

        # Commented-out settings list entries were preserved as comments
        # {"sliding_window_size": 2000, "non_sliding_layers": list(range(0, 50, 2))},
        # {"sliding_window_size": 2000, "non_sliding_layers": list(range(0, 50, 4))},
        # {"sliding_window_size": 2000, "non_sliding_layers": list(range(0, 50, 2)),'force_fa_decode':True},
        # {"sliding_window_size": 2000, "non_sliding_layers": list(range(0, 50, 4)),'force_fa_decode':True},
        # {"sliding_window_size": 4000, "non_sliding_layers": []},
        # {"sliding_window_size": 4000, "non_sliding_layers": [], 'force_fa_decode': True},
        # {"sliding_window_size": 4000, "non_sliding_layers": list(range(0, 50, 2))},
        # {"sliding_window_size": 4000, "non_sliding_layers": list(range(0, 50, 4))},
        # {"sliding_window_size": 4000, "non_sliding_layers": list(range(0, 50, 2)), 'force_fa_decode': True},
        # {"sliding_window_size": 4000, "non_sliding_layers": list(range(0, 50, 4)), 'force_fa_decode': True},

    ]

    for setting in settings_list:
        main(model_path, swaa_config=SWAAConfig(**setting))

