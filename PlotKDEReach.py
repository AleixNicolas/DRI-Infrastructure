import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
import os

# --- Use script directory as origin ---
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "processed_files")
results_dir = os.path.join(script_dir, "results")

# Ensure results folder exists
os.makedirs(results_dir, exist_ok=True)

# --- Color palette ---
color_palette = {
    "10 yr": "#FFD700",
    "100 yr": "#FF7F00",
    "500 yr": "#B22222",

    "DANA_31_10_2024": "#8A2BE2",
    "DANA_03_11_2024": "#FF1493",
    "DANA_05_11_2024": "#00CED1",
    "DANA_06_11_2024": "#32CD32",
    "DANA_08_11_2024": "#1E90FF",

    "Normal Conditions": "#808080"
}

# --- Build file paths ---
shortest_path_files = {
    name: os.path.join(processed_dir, filename)
    for name, filename in {
        "Normal Conditions": "shortest_paths_NP.json",
        "10 yr": "shortest_paths_10.json",
        "100 yr": "shortest_paths_100.json",
        "500 yr": "shortest_paths_500.json",
        "DANA_31_10_2024": "shortest_paths_DANA_31_10_2024.json",
        "DANA_03_11_2024": "shortest_paths_DANA_03_11_2024.json",
        "DANA_05_11_2024": "shortest_paths_DANA_05_11_2024.json",
        "DANA_06_11_2024": "shortest_paths_DANA_06_11_2024.json",
        "DANA_08_11_2024": "shortest_paths_DANA_08_11_2024.json"
    }.items()
}

# --- Prepare data structures ---
plot_data = []
zero_counts = {}
all_counts = {}

layer_names = [
    "DANA_31_10_2024",
    "DANA_03_11_2024",
    "DANA_05_11_2024",
    "DANA_08_11_2024"
]

scenario_names = [
    '31/10/2024',
    '03/11/2024',
    '05/11/2024',
    '08/11/2024',
    'Unperturbed'
]

# --- Load Normal Conditions ---
with open(shortest_path_files["Normal Conditions"], 'r') as f:
    T_NP_dictionary = json.load(f)

# --- Load perturbed scenarios ---
T_P_dictionaries = {}
for name in layer_names:
    with open(shortest_path_files[name], 'r') as f:
        T_P_dictionaries[name] = json.load(f)

# =============================
#       EXTRACT TIME ARRAYS
#     (FIXED NULL HANDLING)
# =============================

for name, results_dict in T_P_dictionaries.items():

    # Raw times (may contain None)
    raw_times = [v["time"] for v in results_dict.values()]

    # Count unreachable (None values)
    unreachable = sum(t is None for t in raw_times)

    # Keep only numeric times
    numeric_times = [t for t in raw_times if isinstance(t, (int, float))]

    # Extract non-zero ( > 0 ) travel times, convert to minutes
    non_zero_times = [t / 60 for t in numeric_times if t > 0]

    # Store counts
    zero_counts[name] = unreachable + numeric_times.count(0)
    all_counts[name] = len(raw_times)

    plot_data.append((name, non_zero_times))

# --- Add Normal Conditions ---
normal_raw = [v["time"] for v in T_NP_dictionary.values()]
normal_unreachable = sum(t is None for t in normal_raw)
normal_numeric = [t for t in normal_raw if isinstance(t, (int, float))]
normal_nonzero = [t / 60 for t in normal_numeric if t > 0]

plot_data.append(("Normal Conditions", normal_nonzero))
all_counts["Normal Conditions"] = len(normal_raw)
zero_counts["Normal Conditions"] = normal_unreachable + normal_numeric.count(0)

# =============================
#           KDE PLOT
# =============================

fig, (ax_kde, ax_bar) = plt.subplots(
    1, 2,
    figsize=(14, 7),
    gridspec_kw={'width_ratios': [2.5, 1]}
)

# Compute global x max
x_vals = np.linspace(
    0,
    max([max(times) if times else 0 for _, times in plot_data]),
    500
)

# --- Plot KDE curves ---
i = 0
for name, times in plot_data:
    if len(times) > 1:
        kde = gaussian_kde(times)
        y = kde(x_vals)

        ratio = len(times) / all_counts[name] if all_counts[name] > 0 else 0
        y_rescaled = y * ratio

        color = color_palette.get(name, 'gray')

        if i < len(scenario_names):
            label = scenario_names[i]
        else:
            label = name

        ax_kde.plot(x_vals, y_rescaled, label=label, color=color)
    i += 1

ax_kde.set_xlim(0, max([max(times) if times else 0 for _, times in plot_data]))
ax_kde.set_ylim(bottom=0)
ax_kde.set_xlabel("Travel Time (minutes)", fontsize=24)
ax_kde.grid(True)
ax_kde.legend(loc='upper right', fontsize=20)
ax_kde.tick_params(axis='both', labelsize=20)
ax_kde.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax_kde.yaxis.set_major_locator(MaxNLocator(nbins=4))

# =============================
#     REACHABILITY BAR CHART
# =============================

scenarios = layer_names
scenario_short_names = [
    '31/10/2024',
    '03/11/2024',
    '05/11/2024',
    '08/11/2024'
]

reachable = [
    (all_counts[n] - zero_counts[n]) / all_counts[n]
    for n in scenarios
]
unreachable = [1 - r for r in reachable]

bar_positions = np.arange(len(scenarios))
ax_bar.barh(bar_positions, unreachable, color='salmon', label='Unreachable')
ax_bar.barh(bar_positions, reachable, left=unreachable,
            color='mediumseagreen', label='Reachable')

ax_bar.set_yticks(bar_positions)
ax_bar.set_yticklabels(scenario_short_names, fontsize=20)
ax_bar.set_xlim(0, 1)
ax_bar.set_xlabel("Fraction of cut routes", fontsize=24)
ax_bar.tick_params(axis='x', labelsize=20)
ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
ax_bar.invert_yaxis()

plt.tight_layout()

# =============================
#         SAVE OUTPUTS
# =============================

plt.savefig(os.path.join(results_dir, 'Travel_times_DANA.pdf'),
            format='pdf', dpi=300)
plt.savefig(os.path.join(results_dir, 'Travel_times_DANA.png'),
            dpi=300)
