import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu
gen = [10, 50, 100, 150, 200, 250, 300]
variants = ["MFEA", "MFEA + Tabu", "MFEA + FLS", "MFEA + ALNS"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

data = {
    "MFEA (TSP)": [100770, 65863, 44645, 38545, 36586, 35519, 34389],
    "MFEA (TRP)": [316.22, 168.95, 108.97, 93.22, 80.62, 68.81, 63.45],
    "MFEA + Tabu (TSP)": [47648, 40819, 38199, 36601, 34801, 33956, 33940],
    "MFEA + Tabu (TRP)": [62.35, 56.85, 52.36, 50.58, 48.46, 47.39, 45.08],
    "MFEA + FLS (TSP)": [36921, 34436, 34325, 34325, 34262, 34106, 33926],
    "MFEA + FLS (TRP)": [50.4, 47.96, 47.96, 46.67, 46.01, 44.13, 44.13],
    "MFEA + ALNS (TSP)": [49525, 37576, 35892, 34832, 34278, 33896, 33896],
    "MFEA + ALNS (TRP)": [573.35, 108.53, 79.54, 71.95, 60.95, 51.02, 49.11]
}

# Vẽ biểu đồ
plt.figure(figsize=(14, 7))
bar_width = 1.2
x = np.arange(len(gen))  # [0, 1, 2, ..., 6]

for i, variant in enumerate(variants):
    tsp_vals = data[f"{variant} (TSP)"]
    trp_vals = data[f"{variant} (TRP)"]
    offset = (i - 1.5) * bar_width
    x_pos = x + offset

    # Vẽ cột cho TRP
    plt.bar(x_pos, trp_vals, width=bar_width, label=f"{variant} (TRP)", color=colors[i], alpha=0.4)

    # Vẽ đường nối cho TSP
    plt.plot(x_pos, tsp_vals, marker='o', linestyle='-', label=f"{variant} (TSP)", color=colors[i], linewidth=2)

plt.xticks(x, gen)
plt.xlabel("Generation")
plt.ylabel("Objective Value")
plt.title("So sánh hiệu quả MFEA và các biến thể trên TSP & TRP")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
