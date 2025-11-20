import matplotlib.pyplot as plt
import json
import os

# Get directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to script location
input_path = os.path.join(script_dir, "processed_files", "R_G.json")
output_path = os.path.join(script_dir, "results", "Risk_Factor.png")

# Load JSON
with open(input_path, 'r') as f:
    R = json.load(f)

# Sort (if needed)
sorted_data = dict(sorted(R.items(), key=lambda item: item[1]))

names = ['High Prob.', 'Med. Prob.', 'Low Prob.', '31/10/2024', '03/11/2024',
         '05/11/2024', '06/11/2024', '08/11/2024']
values = list(R.values())

# Plot
plt.figure(figsize=(8, 5))
plt.bar(names, values, color="Blue", alpha=0.5)
plt.ylim(0, 1)
plt.ylabel('$R_{G}$')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
plt.savefig(output_path)
