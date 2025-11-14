import matplotlib.pyplot as plt
import json

with open("processed_files/R_G.json", 'r') as f:
    R=json.load(f)
sorted_data = dict(sorted(R.items(), key=lambda item: item[1]))
names=['High Prob.', 'Med. Prob.', 'Low Prob.', '31/10/2024', '03/11/2024', '05/11/2024', '06/11/2024', '08/11/2024']
values = list(R.values())

plt.figure(figsize=(8, 5))
plt.bar(names, values, color="Blue", alpha=0.5)

plt.ylim(0, 1)
plt.ylabel('$R_{G}$')
#plt.title('Global Risk Index')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('results/Risk_Factor.png')