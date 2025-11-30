import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
import math
from tqdm import tqdm
import json
from collections import Counter

# Try importing flash_attn, a core dependency for the paper's method
try:
    from flash_attn import flash_attn_func
except ImportError:
    raise ImportError("Please install flash_attn: pip install flash-attn")


def Lazy_ratio_calculation(q, k, v, w_last, w_sink, w_recent):
    """
    Calculates the Lazy Ratio (Log domain), following the paper's formula (1) and Table 1.

    Args:
        q, k, v: Query, Key, Value states [bs, seq_len, num_heads, head_dim]
        w_last: Number of tokens at the end to check
        w_sink: Size of the sink tokens window
        w_recent: Size of the recent tokens window
    """
    # q, k, v shape: [bs, seq_len, num_heads, head_dim]

    # 1. Calculate LSE (Log-Sum-Exp) for the complete Flash Attention (Denominator)
    # return_lse=True returns the softmax normalization factor
    # lse shape: [bs, num_heads, seq_len]
    # Note: Flash Attention requires fp16 or bf16
    out, lse, _ = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)

    # 2. Extract Query states for the last w_last tokens
    # q_last shape: [bs, num_heads, w_last, head_dim] (permuted for matmul)
    q_last = q[:, -w_last:].permute(0, 2, 1, 3)

    # Extract corresponding LSE (Denominator) part (last w_last tokens)
    # lse_last shape: [bs, num_heads, w_last]
    lse_last = lse[:, :, -w_last:]

    # 3. Combine Key states for Sink Tokens and Recent Tokens
    # k shape: [bs, seq_len, num_heads, head_dim]
    k_sink = k[:, :w_sink]
    k_recent = k[:, -w_recent:]
    # k_comb shape: [bs, num_heads, head_dim, num_keys] (permuted for matmul transpose)
    k_comb = torch.cat([k_sink, k_recent], dim=1).permute(0, 2, 3, 1)

    # 4. Calculate Attention Score (Numerator): Query (Last) with Key (Sink + Recent)
    # matmul: [bs, heads, w_last, dim] @ [bs, heads, dim, keys] -> [bs, heads, w_last, keys]
    attn_score = torch.matmul(q_last, k_comb)

    # Scale the attention score (FlashAttn does this internally, but manual matmul needs it)
    head_dim = q.shape[-1]
    attn_score = attn_score / math.sqrt(head_dim)

    # Calculate LogSumExp for the Numerator
    log_numerator = attn_score.logsumexp(dim=-1)

    # 5. Calculate Log Lazy Ratio = Log(Numerator) - Log(Denominator)
    # result shape: [bs, num_heads, w_last]
    log_lazy_ratio = log_numerator - lse_last

    return log_lazy_ratio


def identify_lazy_layers(
        model_path="qwen3-4b-thinking",  # User-specified model
        data_path="longmemeval.parquet",
        num_samples=20,  # Number of samples for frequency count
        w_sink=4,  # Paper setting: 4
        w_recent=1020,  # Paper setting: 1020
        w_last=64,  # Number of trailing tokens for detection
        top_p_ratio=0.5  # Ratio of layers to mark as Lazy Layer
):
    print(f"Loading model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Flash Attention requires float16 or bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        quantization_config=bnb_config if "30b" in model_path.lower() else None,
    )

    # Load data
    print(f"Loading data: {data_path}...")
    df = pd.read_parquet(data_path)

    # Check for required column: 'prompt'
    if 'prompt' not in df.columns:
        raise ValueError("Parquet file is missing the 'prompt' column")

    # Check for required column: 'answer' (or compatible)
    answer_col = None
    for col in ['answer', 'output', 'completion', 'response']:
        if col in df.columns:
            answer_col = col
            break
    if answer_col is None:
        raise ValueError("Parquet file is missing 'answer' (or 'output'/'completion') column")

    # Randomly select num_samples
    if len(df) > num_samples:
        print(f"Randomly sampling {num_samples} from {len(df)} data points...")
        df_subset = df.sample(n=num_samples, random_state=42)  # Use seed for reproducibility
    else:
        df_subset = df

    # Format data using apply_chat_template
    print("Formatting data using apply_chat_template...")
    prompts = []

    # Check if tokenizer has a chat_template
    has_chat_template = tokenizer.chat_template is not None or tokenizer.default_chat_template is not None
    if not has_chat_template:
        print("Warning: Tokenizer does not define a chat_template. Will attempt default/fallback concatenation.")

    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Formatting",mininterval=10):
        prompt_text = str(row['prompt'])
        answer_text = str(row[answer_col])

        # Construct chat history (user question and assistant answer)
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": answer_text}
        ]

        try:
            # apply_chat_template returns the formatted string
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            prompts.append(formatted_text)
        except Exception as e:
            # Fallback to simple concatenation if template fails
            if len(prompts) == 0:
                print(f"apply_chat_template failed: {e}. Falling back to simple concatenation.")
            prompts.append(prompt_text + answer_text)

    # Count how many times each layer is identified as Lazy (Frequency Count method)
    layer_selection_counts = Counter()
    total_valid_samples = 0

    print("Starting Lazy Ratio calculation and frequency counting...")
    model.eval()

    for prompt_idx, prompt in enumerate(tqdm(prompts)):
        # 1. Encode Prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]

        # Skip if sequence is too short for the windows
        if seq_len < (w_sink + w_recent + 10):
            continue

        total_valid_samples += 1
        current_sample_scores = {}  # {layer_idx: lazy_ratio}

        with torch.no_grad():
            outputs = model(
                inputs.input_ids,
                output_hidden_states=True,
                use_cache=False
            )

            for layer_idx, layer in enumerate(model.model.layers):
                # --- Get layer input ---
                hidden_state = outputs.hidden_states[layer_idx]
                hidden_state = layer.input_layernorm(hidden_state)
                attn_module = layer.self_attn

                # --- Project to Q, K, V ---
                bsz, q_len, _ = hidden_state.shape
                num_heads = attn_module.config.num_attention_heads
                num_key_value_heads = attn_module.config.num_key_value_heads
                head_dim = attn_module.head_dim

                query_states = attn_module.q_proj(hidden_state)
                key_states = attn_module.k_proj(hidden_state)
                value_states = attn_module.v_proj(hidden_state)

                query_states = query_states.view(bsz, q_len, num_heads, head_dim)
                key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim)

                if hasattr(attn_module, 'q_norm'):
                    query_states = attn_module.q_norm(query_states)
                if hasattr(attn_module, 'k_norm'):
                    key_states = attn_module.k_norm(key_states)

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                position_ids = torch.arange(q_len, device=query_states.device).unsqueeze(0)
                if hasattr(model.model, "rotary_emb"):
                    rotary_emb = model.model.rotary_emb
                elif hasattr(attn_module, "rotary_emb"):
                    rotary_emb = attn_module.rotary_emb
                else:
                    rotary_emb = getattr(attn_module, "rotary_emb", None)

                cos, sin = rotary_emb(value_states, position_ids)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                if num_key_value_heads < num_heads:
                    key_states = key_states.repeat_interleave(num_heads // num_key_value_heads, dim=2)
                    value_states = value_states.repeat_interleave(num_heads // num_key_value_heads, dim=2)

                # --- Calculate Lazy Ratio ---
                log_ratio = Lazy_ratio_calculation(
                    query_states, key_states, value_states,
                    w_last, w_sink, w_recent
                )

                # Calculate average Lazy Ratio (exp for ratio, then mean)
                avg_lazy_ratio = log_ratio.exp().mean().item()
                current_sample_scores[layer_idx] = avg_lazy_ratio

        # --- Lazy Layer Determination for Current Sample (Local Decision) ---
        # 1. Sort layers by Lazy Ratio (highest first)
        sorted_layers_sample = sorted(current_sample_scores.items(), key=lambda x: x[1], reverse=True)
        # 2. Select the Top-P layers for this sample
        num_layers_sample = int(len(sorted_layers_sample) * top_p_ratio)
        lazy_layers_sample = [x[0] for x in sorted_layers_sample[:num_layers_sample]]

        # 3. Update global counter (Frequency Count)
        layer_selection_counts.update(lazy_layers_sample)

    # --- Final Aggregation (Global Decision) ---
    # Sort by selection frequency
    final_ranking = sorted(layer_selection_counts.items(), key=lambda x: x[1], reverse=True)

    # Select the final Lazy Layers (Top-P from frequency list)
    num_total_layers = len(model.model.layers)
    num_layers_final = int(num_total_layers * top_p_ratio)

    final_lazy_layers = [x[0] for x in final_ranking[:num_layers_final]]

    # --- Calculate Non-Lazy Layers ---
    all_layer_indices = set(range(num_total_layers))
    lazy_layer_set = set(final_lazy_layers)
    non_lazy_layers = sorted(list(all_layer_indices - lazy_layer_set))

    print("\n" + "=" * 30)
    print("Lazy Layer Identification Results (Frequency Voting)")
    print("=" * 30)
    print(f"Valid Samples Used: {total_valid_samples}")
    print(f"Number of Lazy Layers Selected: {num_layers_final}")
    print(f"Lazy Layers (Indices): {sorted(final_lazy_layers)}")
    print(f"Non-Lazy Layers (Indices): {non_lazy_layers}")
    print("-" * 30)
    print("Top 5 Layers (Layer Index: Frequency Count/Total Samples):")
    for idx, count in final_ranking[:5]:
        print(f"Layer {idx}: {count}/{total_valid_samples}")

    # --- Save results to JSON ---
    model_name_safe = os.path.basename(model_path.rstrip("/\\"))
    output_filename = f"{model_name_safe}_lazy_layers_freq.json"

    output_data = {
        "model_path": model_path,
        "method": "frequency_count",
        "total_samples": total_valid_samples,
        "selected_lazy_layers": sorted(final_lazy_layers),
        "non_lazy_layers": non_lazy_layers,
        "layer_frequencies": [
            {"layer_idx": idx, "count": count,
             "frequency": count / total_valid_samples if total_valid_samples > 0 else 0}
            for idx, count in final_ranking
        ]
    }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Detailed frequency statistics saved to: {output_filename}")

    return sorted(final_lazy_layers)


if __name__ == "__main__":
    model_list = [
        "/share/models/Qwen3-4B-Thinking-2507",
        "/share/models/Qwen3-4B-Instruct-2507",
        "/share/models/Meta-Llama-3.1-8B-Instruct",
        "/share/models/Qwen3-30B-A3B-Thinking-2507",
        "/share/models/Qwen3-30B-A3B-Instruct-2507",
    ]

    for model_path in model_list:
        print(f"Processing model: {model_path}")

        lazy_layers = identify_lazy_layers(
            model_path=model_path,
            data_path="../Datasets/fusang_long.parquet",
            num_samples=500,  # Number of samples for frequency count
            w_sink=4,  # Paper setting: 4
            w_recent=1024,  # Paper setting: 1024
            w_last=32,  # Number of trailing tokens for detection
            top_p_ratio=0.5  # Ratio of layers to mark as Lazy Layer
        )
