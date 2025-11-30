import os
import re
import peft
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# Ensure running on CPU to prevent out-of-memory issues on GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def find_latest_checkpoint(base_dir: str) -> str:
    """
    Finds the checkpoint directory with the largest step number in the given base directory.
    E.g., finds 'checkpoint-X' with the maximum X in 'path/to/SFT/training_output/model-name'.
    """

    # Check if the base directory exists
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    max_step = -1
    latest_checkpoint_path = None

    # Regex to match 'checkpoint-number'
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")

    # Iterate through all subdirectories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            match = checkpoint_pattern.match(item)
            if match:
                step = int(match.group(1))
                if step > max_step:
                    max_step = step
                    latest_checkpoint_path = item_path

    if latest_checkpoint_path:
        print(f"✅ Found latest checkpoint: {latest_checkpoint_path} (step: {max_step})")
        return latest_checkpoint_path
    else:
        raise FileNotFoundError(f"Could not find any 'checkpoint-X' folder in: {base_dir}")


def merge_lora_checkpoint(output_dir: str):
    """
    Automatically finds the latest LORA checkpoint in the specified training output directory
    and merges it with the base model.

    Args:
        output_dir (str): The root directory of the LORA training results.
                          Example: "///share/yyj/llm_as_memory/SWA_adaptation/SFT/training_output/Qwen3-4B-Thinking-2507-sft-fusang-all_sli2k_keep100"
    """
    try:
        # 1. Find the path to the latest checkpoint
        lora_weight_path = find_latest_checkpoint(output_dir)

        # Define the path for the merged model
        merged_model_path = lora_weight_path + "-merged"

        print(f"\n--- Starting Merge Operation ---")
        print(f"Source LoRA: {lora_weight_path}")
        print(f"Target Path: {merged_model_path}")

        # Skip if the merged directory already exists
        if os.path.exists(merged_model_path):
            print(f"⚠️ Merge directory already exists. Skipping: {merged_model_path}")
            return

        # 2. Load LoRA configuration
        print("Loading LoRA configuration...")
        lora_config = LoraConfig.from_pretrained(lora_weight_path)
        base_model_path = lora_config.base_model_name_or_path
        print(f"Base Model Path: {base_model_path}")

        # 3. Load the base model
        print("Loading base model...")
        # Force device_map={"":"cpu"} to avoid CUDA environment dependency, and use 'auto' dtype
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype='auto',
            device_map={"": "cpu"}
        )

        # 4. Load LoRA weights
        print("Loading LoRA weights onto base model...")
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_weight_path,
            device_map={"": "cpu"}  # Keep consistent with base_model
        )

        # 5. Merge LoRA weights
        print("Merging LoRA weights and unloading...")
        merged_model = lora_model.merge_and_unload()

        # Print info
        print(f"\nModel Type after merge: {type(merged_model)}")

        # Print model size
        try:
            # .get_memory_footprint() might return 0 in CPU/non-CUDA environment, kept here for potential use
            size_gb = merged_model.get_memory_footprint() / (1024 ** 3)
            print(f"Approximate Model Size (Memory Footprint): {size_gb:.3f} GB")
        except Exception:
            print("Could not calculate memory footprint accurately.")

        # 6. Save the merged model
        print(f"Saving merged model to: {merged_model_path}...")
        merged_model.save_pretrained(merged_model_path)

        # 7. Save Tokenizer
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(merged_model_path)

        print(f'✅ Merge and save completed successfully! Path: {merged_model_path}')

    except Exception as e:
        print(f"❌ An error occurred during the merge process for {output_dir}: {e}")


if __name__ == '__main__':
    # Your list of LORA training output directories
    lora_output_dir_list = [
        "./training_output/Qwen3-4B-Thinking-2507-sft-fusang-all_sli2k_keep100",
    ]

    for output_dir in lora_output_dir_list:
        print(f"\n{'=' * 80}")
        print(f"Processing training output directory: {output_dir}")
        print(f"{'=' * 80}")
        merge_lora_checkpoint(output_dir)