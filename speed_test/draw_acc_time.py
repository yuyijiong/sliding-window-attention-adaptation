import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

# ==========================================
# 1. Raw Data Definition
# ==========================================

# --- Non-SFT Data ---
cols_non_sft = ['id', 'SFT', 'method_count', 'sws', 'nsl', 'keep', 'pre', 'acc_think', 'acc_inst']
data_non_sft = [
    [0, False, 0, float('nan'), "[]", 0, False, 0.73, 0.62],
    [1, False, 0, 2000, "[]", 0, False, 0.032, 0.11],
    [3, False, 1, 2000, "[]", 10, False, 0.16, 0.156],
    [7, False, 2, 2000, "[]", 10, True, 0.382, 0.206],
    #[8, False, 2, 4000, "[]", 10, True, 0.38, 0.244],
    [10, False, 2, 2000, "[]", 100, True, 0.5, 0.178],
    [14, False, 2, 2000, "[1, 3, 5, ...]", 10, False, 0.258, 0.364],
    [17, False, 3, 2000, "[1, 3, 5, ...]", 10, True, 0.65, 0.536],
    [18, False, 3, 2000, "[1, 3, 5, ...]", 100, True, 0.688, 0.506],
    #[19, False, 3, 4000, "[1, 3, 5, ...]", 100, True, 0.73, 0.542],
    [21, False, 3, 2000, "[1, 5, 9, ...]", 10, True, 0.532, 0.314],
]

# --- SFT Data ---
cols_sft = ['id', 'SFT', 'sws', 'nsl', 'keep', 'pre', 'acc_think', 'acc_inst']
data_sft = [
    [0, True, float('nan'), "[]", 0, False, 0.746, 0.634],
    [1, True, 2000, "[]", 0, False, 0.188, 0.238],
    [2, True, 2000, "[]", 0, True, 0.579, 0.42],
    [4, True, 2000, "[1, 3, 5, ...]", 0, False, 0.636, 0.546],
    [6, True, 2000, "[]", 100, True, 0.622, 0.426],
    [8, True, 2000, "[1, 3, 5, ...]", 0, True, 0.732, 0.588],
    [10, True, 2000, "[1, 5, 9, ...]", 0, True, 0.688, 0.47]
]

# --- Time Lookup Table ---
time_lookup_base = [
    {'sws_key': 'Full', 'fa': False, 'nsl_key': 'None', 'duration_s': 3.44},
    {'sws_key': 2000, 'fa': False, 'nsl_key': 'None', 'duration_s': 0.43},
    {'sws_key': 2000, 'fa': False, 'nsl_key': '1in2', 'duration_s': 1.92},
    {'sws_key': 2000, 'fa': True, 'nsl_key': 'None', 'duration_s': 2.01},
    {'sws_key': 2000, 'fa': True, 'nsl_key': '1in2', 'duration_s': 2.72},
    {'sws_key': 2000, 'fa': True, 'nsl_key': '1in4', 'duration_s': 2.37},
    {'sws_key': 4000, 'fa': False, 'nsl_key': 'None', 'duration_s': 0.48},
    {'sws_key': 4000, 'fa': False, 'nsl_key': '1in2', 'duration_s': 1.94},
    {'sws_key': 4000, 'fa': True, 'nsl_key': 'None', 'duration_s': 2.06},
    {'sws_key': 4000, 'fa': True, 'nsl_key': '1in2', 'duration_s': 2.77},
    {'sws_key': 4000, 'fa': True, 'nsl_key': '1in4', 'duration_s': 2.41},
]


# ==========================================
# 2. Data Integration and Processing
# ==========================================

def get_nsl_key(nsl_str):
    if nsl_str == "[]" or pd.isna(nsl_str): return 'None'
    if "1, 3" in nsl_str: return '1in2'
    if "1, 5" in nsl_str: return '1in4'
    return 'None'


def get_sws_key(val):
    return 'Full' if pd.isna(val) else val


def build_acc_table(acc_col_name):
    result_table = []

    df1 = pd.DataFrame(data_non_sft, columns=cols_non_sft)
    df1['is_sft'] = False
    df2 = pd.DataFrame(data_sft, columns=cols_sft)
    df2['is_sft'] = True
    df_all = pd.concat([df1, df2], ignore_index=True)

    for _, row in df_all.iterrows():
        if pd.isna(row[acc_col_name]): continue

        s_key = get_sws_key(row['sws'])
        n_key = get_nsl_key(row['nsl'])
        f_key = row['pre']

        found_time = None
        for t_entry in time_lookup_base:
            if t_entry['sws_key'] == s_key and t_entry['nsl_key'] == n_key and t_entry['fa'] == f_key:
                found_time = t_entry['duration_s']
                break

        if found_time is not None:
            result_table.append({
                'id': row['id'],
                'is_sft': row['is_sft'],
                'time': found_time,
                'acc': row[acc_col_name] * 100,
                'sws': row['sws'],
                'nsl': row['nsl'],
                'keep': row['keep'],  # Keep data, but not used for Config distinction
                'pre': row['pre']
            })

    return result_table


time_acc_think = build_acc_table('acc_think')
time_acc_instruct = build_acc_table('acc_inst')


# ==========================================
# 3. Color Mapping and Filtering Logic
# ==========================================

# 1. Extract Unique Config Identifier (omit keep)
def get_config_key(p):
    """
    Unique Config Key: (sws, nsl, pre).
    'keep' is omitted as different 'keep' values are considered the same config.
    """
    sws_val = p['sws'] if pd.notna(p['sws']) else "Full"
    return (sws_val, p['nsl'], p['pre'])

# Set font (Removed: Assuming default or external setting)

# 2. Prepare Plotting Data
for current_data in [time_acc_think, time_acc_instruct]:

    acc_label = "Accuracy (%)"

    # 3. Filtering Logic: Group by (is_sft, Config) and select max Acc
    final_points = []
    groups = defaultdict(list)

    for p in current_data:
        # Key includes is_sft to separate SFT and Non-SFT data
        # Key includes get_config_key(p) to group same parameters (ignoring keep)
        key = (p['is_sft'], get_config_key(p))
        groups[key].append(p)

    for key, group in groups.items():
        # Select the entry with the highest accuracy within the same group (same config, different keep)
        best = max(group, key=lambda x: x['acc'])
        final_points.append(best)

    # 4. Generate Color Map (based on Config Key)

    # Custom sort key
    def config_sort_key(cfg):
        sws_val, nsl_val, pre_val = cfg

        # 1. Window Size: Full first (-1), others by numerical value
        if sws_val == "Full":
            s_rank = -1
        else:
            s_rank = float(sws_val)

        # 2. FA Layers: 0 (None) -> 1/4 (1,5) -> 1/2 (1,3)
        if nsl_val == "[]" or pd.isna(nsl_val) or nsl_val == 'None':
            n_rank = 0
        elif "1, 5" in nsl_val: # 1in4
            n_rank = 1
        elif "1, 3" in nsl_val: # 1in2
            n_rank = 2
        else:
            n_rank = 3 # Other cases

        # 3. FA Decode: False (0) -> True (1)
        p_rank = 1 if pre_val else 0

        return (s_rank, n_rank, p_rank)

    # Apply sorting
    unique_configs = sorted(list(set(get_config_key(p) for p in final_points)),
                            key=config_sort_key)

    # Color pool
    tab20c_indices = []
    for shade in range(4):
        for hue in range(0, 20, 4):
            tab20c_indices.append(hue + shade)

    # Config -> Color mapping
    config_color_map = {}
    for i, cfg_key in enumerate(unique_configs):
        color_idx = tab20c_indices[i % 20]
        config_color_map[cfg_key] = plt.cm.tab20(color_idx)

    # ==========================================
    # 4. Plotting
    # ==========================================
    plt.figure(figsize=(8,8))

    # Plot scatter points
    for p in final_points:
        cfg_key = get_config_key(p)
        c = config_color_map[cfg_key]

        marker = 'o' if p['is_sft'] else 's'
        plt.scatter(p['time'], p['acc'],
                    color=c,
                    marker=marker, s=150,
                    edgecolor='black', linewidth=0.1, alpha=0.9, zorder=3)

    # Draw connecting lines
    for is_sft in [True, False]:
        # Find Full Attention point
        p_full = next((p for p in final_points if p['is_sft'] == is_sft and pd.isna(p['sws'])), None)

        # Find 2000 Window, No NSL, No FA point (Original ID 1 or 3)
        p_base = next((p for p in final_points
                       if p['is_sft'] == is_sft
                       and p['sws'] == 2000
                       and (p['nsl'] == "[]" or pd.isna(p['nsl']))
                       and p['pre'] == False), None)

        if p_full and p_base:
            line_style = ':' if is_sft else '--'
            plt.plot([p_full['time'], p_base['time']], [p_full['acc'], p_base['acc']],
                     color='gray', linestyle=line_style, linewidth=2 if is_sft else 1.5, zorder=0)

    #plt.title(f"Accuracy vs Time ({'Thinking' if current_data == time_acc_think else 'Non-thinking'})", fontsize=18)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel(acc_label, fontsize=16)

    # [Modification] Increase tick font size
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # 5. Table Generation (Modified Section)
    # ==========================================

    cell_text = []
    row_colors = []

    # Since unique_configs is already sorted, the table row order will be correct
    for cfg_key in unique_configs:
        c = config_color_map[cfg_key]
        sws_val, nsl_val, pre_val = cfg_key  # Unpack (omit keep)

        # Formatting
        sws_str = str(int(sws_val) // 1000) + "k" if sws_val != "Full" else "Full"

        # [Modification] FA Layers display logic
        if "1, 3" in nsl_val:
            nsl_str = "1/2"
        elif "1, 5" in nsl_val:
            nsl_str = "1/4"
        else:
            nsl_str = "0"

        # [Modification] FA Decode display logic
        pre_str = "True" if pre_val else "False"

        # Remove Keep column
        cell_text.append(["●", sws_str, nsl_str, pre_str])
        row_colors.append(c)

    columns = ["Color", "Window", "FA Layers", "FA Decode"]

    plt.subplots_adjust(bottom=0.25)
    the_table = plt.table(cellText=cell_text, colLabels=columns,
                          loc='bottom', cellLoc='center', edges='open',
                          bbox=[0.15, -0.55, 0.7, 0.4])

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(15)
    the_table.scale(1, 1.6)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0: cell.set_text_props(fontweight='bold')
        if row > 0 and col == 0: cell.set_text_props(color=row_colors[row - 1], fontsize=16)

    plt.text(0.98, 0.05, "●: w/ SFT | ■:  w/o SFT",
             transform=plt.gca().transAxes, ha='right', fontsize=16,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig('acc_time_{}.png'.format("thinking" if current_data == time_acc_think else "non_thinking")
                , bbox_inches='tight')
    plt.show()