import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

color_palette = {
    # yr values — warm and spaced
    "10 yr": "#FFD700",    # Gold
    "100 yr": "#FF7F00",   # Dark Orange
    "500 yr": "#B22222",   # Firebrick 

    # DANA values — distinct, avoiding orange/red hues
    "DANA_31_10_2024": "#8A2BE2",  # Blue-Violet
    "DANA_03_11_2024": "#FF1493",  # Deep Pink
    "DANA_05_11_2024": "#00CED1",  # Dark Turquoise
    "DANA_06_11_2024": "#32CD32",  # Lime Green
    "DANA_08_11_2024": "#1E90FF",  # Dodger Blue

    # Normal condition
    "Normal Conditions": "#808080"  # Grey
}

output_dir = "processed_files"
shortest_path_files = {
    "Normal Conditions": f"{output_dir}/shorthest_paths_NP.json", 
    "10 yr": f"{output_dir}/shorthest_paths_10.json",
    "100 yr": f"{output_dir}/shorthest_paths_100.json",
    "500 yr": f"{output_dir}/shorthest_paths_500.json",
    "DANA_31_10_2024": f"{output_dir}/shorthest_paths_DANA_31_10_2024.json",
    "DANA_03_11_2024": f"{output_dir}/shorthest_paths_DANA_03_11_2024.json",
    "DANA_05_11_2024": f"{output_dir}/shorthest_paths_DANA_05_11_2024.json",
    "DANA_06_11_2024": f"{output_dir}/shorthest_paths_DANA_06_11_2024.json",
    "DANA_08_11_2024": f"{output_dir}/shorthest_paths_DANA_08_11_2024.json"
}

# --- Prepare data ---
plot_data = []
zero_counts = {}
all_counts = {}
layer_names = ["DANA_31_10_2024","DANA_03_11_2024","DANA_05_11_2024","DANA_08_11_2024"]
scenario_names=['31/10/2024', '03/11/2024', '05/11/2024','08/11/2024','Unperturbed']

with open(shortest_path_files["Normal Conditions"],'r') as f:
    T_NP_dictionary=json.load(f)

T_P_dictionaries = {}
for i, name in enumerate(layer_names):
    with open(shortest_path_files[name], 'r') as f:
        T_P_dictionaries[name] = json.load(f)

for name, results_dict in T_P_dictionaries.items():
    if name in layer_names:
        travel_times = [v[1] for v in results_dict.values()]
        non_zero_times = [t / 60 for t in travel_times if t > 0]
        all_times = [t / 60 for t in travel_times]
        zero_counts[name] = travel_times.count(0)
        all_counts[name] = len(travel_times)
        plot_data.append((name, non_zero_times))

# Add normal scenario
normal_travel_times = [v[1] for v in T_NP_dictionary.values()]
normal_non_zero = [t / 60 for t in normal_travel_times if t > 0]
normal_total = len(normal_travel_times)
plot_data.append(("Normal Conditions", normal_non_zero))
all_counts["Normal Conditions"] = normal_total
zero_counts["Normal Conditions"] = normal_travel_times.count(0)

# --- KDE Plot ---
fig, (ax_kde, ax_bar) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2.5, 1]})

x_vals = np.linspace(0, max([max(times) if times else 0 for _, times in plot_data]), 500)

i = 0
for name, times in plot_data:
    if len(times) > 1:
        kde = gaussian_kde(times)
        y = kde(x_vals)
        ratio = len(times) / all_counts[name] if all_counts[name] > 0 else 0
        y_rescaled = y * ratio
        color = color_palette.get(name, 'gray')  # fallback to gray if not defined
        ax_kde.plot(x_vals, y_rescaled, label=scenario_names[i], color=color)
    i += 1

ax_kde.set_xlim(0, max([max(times) if times else 0 for _, times in plot_data]))
ax_kde.set_ylim(bottom=0)

ax_kde.set_xlabel("Travel Time (minutes)", fontsize=24)
ax_kde.grid(True)
ax_kde.legend(loc='upper right', fontsize=20)
ax_kde.tick_params(axis='both', labelsize=20)

ax_kde.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
ax_kde.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))

# --- Reachability Bar Chart ---
scenarios = layer_names
scenario_names=['31/10/2024', '03/11/2024', '05/11/2024', '08/11/2024']
reachable = [(all_counts[n] - zero_counts[n]) / all_counts[n] if all_counts[n] > 0 else 0 for n in scenarios]
unreachable = [1 - r for r in reachable]

bar_positions = np.arange(len(scenarios))
ax_bar.barh(bar_positions, unreachable, color='salmon', label='Unreachable')
ax_bar.barh(bar_positions, reachable, left=unreachable, color='mediumseagreen', label='Reachable')

ax_bar.set_yticks(bar_positions)
ax_bar.set_yticklabels(scenario_names, fontsize=20)
ax_bar.set_xlim(0, 1)
ax_bar.set_xlabel("Fraction of cut routes", fontsize=24)
ax_bar.tick_params(axis='x', labelsize=20)
ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

ax_bar.invert_yaxis()

plt.tight_layout()

plt.savefig('results/Travel_times_DANA.pdf', format='pdf', dpi=300)
plt.savefig('results/Travel_times_DANA.png', dpi=300)
