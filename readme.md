# Sliding Window Attention Adaptation
This repository contains code and resources for the paper titled "Sliding Window Attention Adaptation"

## Installation
1. Make sure you install the requirements:
```
vllm >= 0.11.0,<0.12.0
transformers >= 4.57.0
```

2. Install the customized flash-attention package:
```bash
git clone https://github.com/yuyijiong/flash-attention-SWAA
cd flash-attention-SWAA
bash install.sh
```
* CUDA >=12.8 is recommended.
* You need to compile from source, so it may take some time.
* It will overwrite any existing flash-attn installation in your environment.
* Note that if nvcc compilation fails, you can try to set a smaller `MAX_JOBS` in `install.sh`, e.g., `export MAX_JOBS=4`.

3. This repository doesn't need to be installed as a package. Just clone it to your local machine and run (or import) the scripts directly.

## Usage

### Sliding Window Attention Adaptation (SWAA) Config
SWAAConfig usually has 4 parameters to set:
- `sliding_window_size`: the size of the window size for sliding window attention. Default is `None`, which means using full attention.
- `keep_first`: the number of initial tokens to always keep in attention. Default is `0`, which means no tokens are always kept.
- `force_fa_decode`: whether to force the model to use full attention during decoding. Default is `False`.
- `non_sliding_layers`: the list of layer indices that do not use sliding window attention. Default is `[]`, which means all layers use sliding window attention.

### Core code
- The core code for SWAA is in the `Patch` folder. It uses monkey patching to modify the attention mechanism of transformers and vLLM.
- Currently we only support `Qwen3, Qwen2, Qwen3MoE, Llama` models

1. To use transformers (HuggingFace) with SWAA:
```python
# include "Patch" folder 
import sys
sys.path.append("./sliding-window-attention-adaptation/Patch")
# before running your code, import the function from hack_hf_swaa.py to patch transformers
from hack_hf_swaa import hack_hf_swaa, SWAAConfig
hack_hf_swaa(training=False)
...
# then you can load the model as usual
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map={"": device_id},
                                             dtype="bfloat16",
                                             trust_remote_code=True,
                                             attn_implementation="flash_attention_2",
                                             ).eval()
...
# finally, set your SWAA config and add it to the model's config,for example:
swaa_config = SWAAConfig(
    sliding_window_size=2048,
    keep_first=100,
    force_fa_decode=True,
    non_sliding_layers=[1,3,5,7,9,11],
)
model.config.swaa_config=swaa_config
...
# now you can use the model as usual
outputs = model.generate(**inputs)
```

2. To use vLLM offline inference with SWAA:
```python
# include "Patch" folder 
import sys
sys.path.append("./sliding-window-attention-adaptation/Patch")
# before running your code, import the function from hack_vllm_0110_swaa.py to patch vLLM
from hack_vllm_0110_swaa import hack_vllm_swaa
hack_vllm_swaa()
...
# then set your SWAA config
swaa_config = SWAAConfig(
    sliding_window_size=2048,
    keep_first=100,
    force_fa_decode=True,
    non_sliding_layers=[1,3,5,7,9,11],
)
...
# finally, initialize the LLM with the 'swaa_config' parameter, for example:
llm = LLM(model=model_path,
          dtype="float16",
          tensor_parallel_size=1,
          enforce_eager=True,  # for SWAA, we must enforce eager mode
          quantization=None,
          swaa_config=swaa_config,
)
...
# now you can use the model as usual
outputs = llm.generate(prompts=batch_prompts, sampling_params=sampling_params)
```


3. To use vLLM server with SWAA, for example, run:
```bash
# cd into "Patch" folder
cd ./sliding-window-attention-adaptation/Patch

# start server with the customized serve_swaa.py
python serve_swaa.py \
    --tensor-parallel-size 1 \
    --port 5000 \
    --served-model-name qwen-4b-swaa \
    --model "YOUR_PATH/Qwen3-4B-Thinking-2507" \
    --max-model-len 50000 \
  # below are SWAA specific arguments
  --sliding-window-size 2048 \
  --keep-first 100 \
  --force-fa-decode True \
  --non-sliding-layers 1,3,5,7,9,11
  
# then you can send requests to the server as usual, for example, see ./Eval/test_vllm_server.py 
cd ../Eval
python test_vllm_server.py
```

## Datasets
1. `./Datasets/fusang_long.parquet`: The training dataset for long-context SFT. [Download link](https://huggingface.co/datasets/yuyijiong/fusang-v1-filtered/)
2. `./Datasets/longmemeval_24k.parquet`: The benchmark dataset for evaluation. [Download link](https://huggingface.co/datasets/yuyijiong/LongMemEval_24k)
3. `./Datasets/longbenchv2_qa.parquet`: Another dataset for evaluation. [Download link](https://huggingface.co/datasets/yuyijiong/LongMemEval_24k)

## Evaluation
1. To run evaluation, refer to `./Eval/eval_swaa.py`. You can modify the parameters or code in the "main" part of the script as needed.
2. It is recommended to write your model path and SWAA configurations in json files, like those in `./Eval/settings_list/`. Then let `./Eval/eval_swaa.py` read them.
3. Any dataset can be used for evaluation, but should have at least the 3 fields: 
   - `prompt`: the whole input prompt, including the long context and the task instruction.
   - `question`: the question or the task instruction only, which should be relatively short.
   - `answer`: the reference answer for evaluation, which should be relatively short.
4. The evaluation results will be saved as a json file in `./Eval/eval_output/`

## Fine-tuning
1. `./SFT/self_distill_data.py` can be used to generate self-distillation data.
2. To run fine-tuning, refer to `./SFT/sft_swaa.py`. You can modify the parameters or code in the "main" part of the script as needed.

## Efficiency Test
1. To run efficiency test on vllm, refer to `./speed_test/time_test_vllm.sh`. You can modify the parameters in the script as needed.
2. `./speed_test/parse_time_json.py` can collect the time test results and print them in markdown table format.
3. To run efficiency test on HF transformers, refer to `./speed_test/speed_test_hf.py`. 

## LightTransfer
1. To run the baseline method LightTransfer, use `./LightTransfer/get_lazy_ratio.py`. You can modify the parameters or code in the "main" part of the script as needed.

## To-do
- [ ] Use vllm plugin system (instead of monkey patching) to integrate SWAA more flexibly.
- [ ] Implement with Sglang
- [ ] Implement with FlashInfer
- [ ] Implement the memory release mechanism corresponding to KV cache discarding.
- [ ] More model support, e.g., Mistral, etc.