import os
import json
import re
import pandas as pd


def process_benchmark_data(root_dir):
    data_list = []

    # Check if root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return pd.DataFrame()

    # Print folder count
    folder_count = len([name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))])
    print(f"Found {folder_count} folders to process.")

    # Iterate over all items in the root directory
    for folder_name in os.listdir(root_dir):
        print("Processing folder:", folder_name)
        folder_path = os.path.join(root_dir, folder_name)

        # Ensure it is a directory
        if not os.path.isdir(folder_path):
            continue

        # --- 1. Parse config parameters from folder name ---

        # Extract sliding_window_size (defaults to "Full" if not present)
        sw_match = re.search(r"sliding_window_size=(\d+)", folder_name)
        sliding_window_size = int(sw_match.group(1)) if sw_match else "Full"

        # Extract keep_first (defaults to 0 if not present)
        kf_match = re.search(r"keep_first=(\d+)", folder_name)
        keep_first = int(kf_match.group(1)) if kf_match else 0

        # Extract force_fa_decode
        fa_match = re.search(r"force_fa_decode=(true|false)", folder_name, re.IGNORECASE)
        force_fa_decode = fa_match.group(1) if fa_match else None

        # Extract non_sliding_layers
        # Logic: Extract content between "non_sliding_layers=" and "-BENCH"
        layer_match = re.search(r"non_sliding_layers=(.*?)-", folder_name)
        non_sliding_layers = layer_match.group(1) if layer_match else None

        # If parsed layers is an empty string, label as "None"
        if non_sliding_layers == "":
            non_sliding_layers = "None"

        # --- 2. Read and process JSON file ---
        json_file_path = os.path.join(folder_path, "summary.json")

        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                # Handle case where JSON is a list containing a dict (as in the example)
                record = content[0] if isinstance(content, list) else content

                # Get metrics (ms)
                ttft_ms = record.get("mean_ttft_ms", 0)
                tpot_ms = record.get("mean_tpot_ms", 0)
                duration= record.get("duration", 0)
                total_token_throughput = record.get("total_token_throughput", 0)

                # Convert to seconds (s) and round to 2 decimal places
                ttft_s = round(ttft_ms / 1000, 2)
                tpot_s = round(tpot_ms / 1000, 2)
                duration_s = round(duration / 1000, 2)
                total_token_throughput = round(total_token_throughput / 1000, 2)  # Convert to thousands (k)

                # Add to data list
                data_list.append({
                    "sliding_window_size": sliding_window_size,
                    "keep_first": keep_first,
                    "force_fa_decode": force_fa_decode,
                    "non_sliding_layers": non_sliding_layers,
                    "mean_ttft_s": ttft_s,
                    "mean_tpot_s": tpot_s,
                    "duration_s": duration_s,
                    "total_token_throughput_k": total_token_throughput
                })

            except Exception as e:
                print(f"Error reading file for {folder_name}: {e}")
        else:
            print(f"Warning: summary.json not found in {folder_name}")

    # --- 3. Generate Pandas DataFrame ---
    df = pd.DataFrame(data_list)

    # Optional: Sort by sliding_window_size for better viewing (removed: sorting line)

    return df


def normalize_data(df):
    # 1. Find the Baseline row
    # Logic: Find rows where 'sliding_window_size' is "Full" or NaN (i.e., the folder without sliding_window_size specified)
    baseline_mask = (df['sliding_window_size'] == 'Full') | (df['sliding_window_size'].isna())

    if baseline_mask.sum() == 0:
        print("Error: Baseline data (sliding_window_size=Full/None) not found. Cannot normalize.")
        return df

    # Get baseline values (default to first if multiple, usually there is only one baseline)
    baseline_row = df.loc[baseline_mask].iloc[0]
    base_ttft = baseline_row['mean_ttft_s']
    base_tpot = baseline_row['mean_tpot_s']
    best_duration = baseline_row['duration_s']
    best_throughput = baseline_row['total_token_throughput_k']

    print(f"--- Normalization Baseline ---")
    print(f"Baseline TTFT: {base_ttft} s")
    print(f"Baseline TPOT: {base_tpot} s")
    print("-" * 30)

    # 2. Calculate Normalized values
    # Add new columns norm_ttft and norm_tpot
    df['norm_ttft'] = df['mean_ttft_s'] / base_ttft
    df['norm_tpot'] = df['mean_tpot_s'] / base_tpot
    # Calculate normalized duration and throughput
    df['norm_duration'] = df['duration_s'] / best_duration
    df['norm_throughput'] = df['total_token_throughput_k'] / best_throughput

    # 3. Format/Round decimals (Optional: 4 decimals to observe small differences)
    df['norm_ttft'] = df['norm_ttft'].round(4)
    df['norm_tpot'] = df['norm_tpot'].round(4)
    df['norm_duration'] = df['norm_duration'].round(4)
    df['norm_throughput'] = df['norm_throughput'].round(4)

    # Calculate norm_avg
    df['norm_avg'] = ((df['norm_ttft'] + df['norm_tpot']) / 2).round(4)

    # Drop original data
    df = df.drop(columns=['mean_ttft_s', 'mean_tpot_s', 'duration_s', 'total_token_throughput_k'])

    return df

if __name__ == '__main__':

    # --- Execute Code ---
    # Replace 'dir' with your actual folder path
    root_directory = 'vllm_bench_results/20251124_234214'
    df_result = process_benchmark_data(root_directory)
    print("Raw Results:")
    print(df_result.to_markdown(index=False))

    df_normalized = normalize_data(df_result)
    # Print results
    print("\nNormalized Results:")
    print(df_normalized.to_markdown(index=False))