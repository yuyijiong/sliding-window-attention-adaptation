import pandas as pd

# Count the number of active SWAA methods applied
def count_methods(row):
    count = 0
    # If non_sliding_layers is not empty, increment method count
    if row['non_sliding_layers']:
        count += 1
    # If keep_first > 0, increment method count
    if row['keep_first'] > 0:
        count += 1
    # If force_fa_decode is True, increment method count
    if row['force_fa_decode']:
        count += 1
    return count

# Extract the base model name
def extract_model_name(model_path):
    if "qwen3-4b-thinking" in model_path.lower():
        return "Qwen3-4B-Thinking"
    elif "qwen3-4b-instruct" in model_path.lower():
        return "Qwen3-4B-Instruct"
    elif "qwen3-30b-a3b-thinking" in model_path.lower():
        return "Qwen3-30B-A3B-Thinking"
    elif "qwen3-30b-a3b-instruct" in model_path.lower():
        return "Qwen3-30B-A3B-Instruct"
    elif "llama-3.1-8b-instruct" in model_path.lower():
        return "Llama3.1-8B-Instruct"
    else:
        raise Exception("Unknown model name in path: " + model_path)

if __name__ == '__main__':
    model_family="qwen3-30b" # Options: "qwen3-4b", "qwen3-30b", "llama3.1-8b"
    thinking_same_row=True # If True, merge 'thinking' and 'instruct' results into the same row

    # Load evaluation results
    df=pd.read_json("./eval_output/result_ruler_niah_multiquery_128k.jsonl",lines=True)

    # Use auto_prefill_slide to fill NaN values in force_fa_decode
    if "force_fa_decode" in df.columns and "auto_prefill_slide" in df.columns:
        df['force_fa_decode']=df['force_fa_decode'].fillna(df['auto_prefill_slide'])
    elif "auto_prefill_slide" in df.columns:
        df['force_fa_decode']=df['auto_prefill_slide']

    # Convert force_fa_decode to boolean (1/True -> True, 0/False -> False). Drop rows with other types (e.g., list)
    def convert_force_fa_decode(value):
        if isinstance(value,list):
            return None

        if value == 1 or value is True:
            return True
        elif value == 0 or value is False:
            return False
        else:
            return None

    df['force_fa_decode']=df['force_fa_decode'].apply(convert_force_fa_decode)
    df=df[df['force_fa_decode'].notna()]

    # If sliding_window_size is NaN, force force_fa_decode to False
    df.loc[df['sliding_window_size'].isna(), 'force_fa_decode'] = False

    # Convert non_sliding_layers to tuple for hashing
    df['non_sliding_layers']=df['non_sliding_layers'].apply(tuple)

    # Remove duplicates: keep the last record if 'model_name', 'sliding_window_size', 'non_sliding_layers', 'keep_first', and 'force_fa_decode' are the same
    df=df.drop_duplicates(subset=['model_name', 'sliding_window_size', 'non_sliding_layers', 'keep_first', 'force_fa_decode'], keep='last')

    # Calculate method count
    df['method_count']=df.apply(count_methods, axis=1)

    # Extract base model name
    df['model']=df['model_name'].apply(extract_model_name)
    # Keep only the specified model_family
    df=df[df['model'].str.contains(model_family,case=False)]

    # Add SFT column: True if 'sft' is in model_name, False otherwise
    df['SFT'] = df['model_name'].apply(lambda x: 'sft' in x.lower())

    # Keep only necessary columns
    df = df[['model','SFT','method_count', 'sliding_window_size', 'non_sliding_layers', 'keep_first', 'force_fa_decode', 'overall_accuracy']]


    if thinking_same_row:
        # Merge 'thinking' and 'instruct' results into the same row
        df_thinking = df[df['model'].str.contains('thinking', case=False)].copy()
        df_instruct = df[df['model'].str.contains('instruct', case=False)].copy()

        # Rename accuracy columns
        df_thinking = df_thinking.rename(columns={'overall_accuracy': 'accuracy_thinking'})
        df_instruct = df_instruct.rename(columns={'overall_accuracy': 'accuracy_instruct'})

        # Remove -thinking and -instruct suffixes from model name
        df_thinking['model'] = df_thinking['model'].str.replace('-thinking', '', case=False)
        df_instruct['model'] = df_instruct['model'].str.replace('-instruct', '', case=False)

        # Merge DataFrames
        df_merged = pd.merge(df_thinking, df_instruct, on=["model",'sliding_window_size', 'non_sliding_layers', 'keep_first', 'force_fa_decode', 'method_count','SFT'], how='left')

    else:
        df_merged = df

    # Sort the results
    df_sorted = df_merged.sort_values(
        by=['model', 'SFT', 'method_count', 'sliding_window_size','non_sliding_layers', 'force_fa_decode', 'keep_first'],
        ascending=[True, True, True, True, True, True, True],
        na_position="first"
    )


    # Format non_sliding_layers: convert to list and truncate to the first 3 elements for display
    def format_non_sliding_layers(layers):
        layers=list(layers)
        if len(layers) > 3:
            return str(layers[:3])[:-1] + ", ...]"
        else:
            return str(layers)

    df_sorted['non_sliding_layers'] = df_sorted['non_sliding_layers'].apply(format_non_sliding_layers)
    df_sorted.reset_index(drop=True, inplace=True)

    # Print as Markdown table
    print(df_sorted.to_markdown(index=True))