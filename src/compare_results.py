"""
compare_results.py
Load metrics for:
  1. Baseline (SingleRes CNN)
  2. Multi-Resolution 3-Branch (MultiResAttentionNet)
  3. SOTA Multi-Feature Differential Attention (MultiFeatureCoordNet)
Print a comprehensive comparison table and save a 4-panel comparison visualization.

Usage:
    python compare_results.py
"""
import json
import os
import matplotlib.pyplot as plt

multires_path = "results/multires_metrics.json"
baseline_path = "results/baseline_metrics.json"
multifeature_path = "results/multifeature_metrics.json"

models_data = {}
if os.path.exists(baseline_path):
    with open(baseline_path) as f: models_data["Baseline (Single-Res)"] = json.load(f)
if os.path.exists(multires_path):
    with open(multires_path) as f: models_data["Multi-Res (3-Branch)"] = json.load(f)
if os.path.exists(multifeature_path):
    with open(multifeature_path) as f: models_data["Multi-Feature (Diff-Attn)"] = json.load(f)

print("\n" + "=" * 95)
print(f"{'EVALUATION BENCHMARK: ARCHITECTURE COMPARISON ON ESC-50':^95}")
print("=" * 95)

headers = f"{'Metric':<25}" + "".join([f"{name:<25}" for name in models_data.keys()])
print(headers)
print("-" * 95)

# 1. Accuracy
row_acc = f"{'Test Accuracy':<25}"
for m in models_data.values():
    acc = m.get('mean_test_accuracy', m.get('test_accuracy', 0.0)) * 100
    row_acc += f"{acc:.2f}%{'':<19}"
print(row_acc)

# 2. Macro F1
row_f1 = f"{'Macro F1 Score':<25}"
for m in models_data.values():
    f1 = m.get('mean_test_macro_f1', m.get('test_macro_f1', 0.0))
    row_f1 += f"{f1:.4f}{'':<19}"
print(row_f1)

# 3. Parameters
row_params = f"{'Parameters':<25}"
for m in models_data.values():
    p = m.get('num_params', 0) / 1e6
    row_params += f"{p:.2f} M{'':<19}"
print(row_params)

# 4. Latency
row_lat = f"{'Batched Latency':<25}"
for m in models_data.values():
    lat = m.get('latency_ms_per_sample', 0.0)
    row_lat += f"{lat:.2f} ms/sample{'':<10}"
print(row_lat)

print("=" * 95 + "\n")

# Generate 4-panel comparison visualization
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
names = list(models_data.keys())
short_names = ["Baseline\n(Single-Res)", "Multi-Res\n(3-Branch)", "Multi-Feature\n(Diff-Attn)"] if len(names) == 3 else names
colors = ["#718096", "#e53e3e", "#2b6cb0"] if len(names) == 3 else ["#718096", "#2b6cb0"]

accs = [m.get('mean_test_accuracy', m.get('test_accuracy', 0.0)) * 100 for m in models_data.values()]
f1s = [m.get('mean_test_macro_f1', m.get('test_macro_f1', 0.0)) for m in models_data.values()]
params = [m.get('num_params', 0) / 1e6 for m in models_data.values()]
latencies = [m.get('latency_ms_per_sample', 0.0) for m in models_data.values()]

# 1. Accuracy
bars0 = axes[0].bar(short_names, accs, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
axes[0].set_title("Test Accuracy (%)", fontsize=11, fontweight="bold")
axes[0].set_ylim(0, 100)
for bar in bars0:
    y = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, y + 2, f"{y:.1f}%", ha="center", fontsize=10, fontweight="semibold")

# 2. Macro F1
bars1 = axes[1].bar(short_names, f1s, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
axes[1].set_title("Macro F1 Score", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 1.0)
for bar in bars1:
    y = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, y + 0.03, f"{y:.3f}", ha="center", fontsize=10, fontweight="semibold")

# 3. Parameters (Millions)
bars2 = axes[2].bar(short_names, params, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
axes[2].set_title("Parameters (M)", fontsize=11, fontweight="bold")
axes[2].set_ylim(0, max(params) * 1.25)
for bar in bars2:
    y = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2, y + 0.8, f"{y:.1f}M", ha="center", fontsize=10, fontweight="semibold")

# 4. Latency (ms/sample)
bars3 = axes[3].bar(short_names, latencies, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
axes[3].set_title("Latency (ms / sample)", fontsize=11, fontweight="bold")
axes[3].set_ylim(0, max(latencies) * 1.25)
for bar in bars3:
    y = bar.get_height()
    axes[3].text(bar.get_x() + bar.get_width()/2, y + (max(latencies) * 0.03), f"{y:.2f} ms", ha="center", fontsize=10, fontweight="semibold")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/comparison.png", dpi=200)
print("Saved 4-panel comparison visualization to results/comparison.png")
