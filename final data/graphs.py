import matplotlib.pyplot as plt
import numpy as np

runs = {
    r"C:\Users\idine\PycharmProjects\Deep Hedging\final data\CNN, 1e-3, 1000, 6c.npy": "λ = 1e-3",
    r"C:\Users\idine\PycharmProjects\Deep Hedging\final data\CNN, l1, 1000, 6.npy": "λ = 1",
}

data, stats = {}, {}          # label → ndarray  / summary

for path, label in runs.items():
    arr          = np.load(path)
    data[label]  = arr
    stats[label] = dict(mu=arr.mean(),
                        sigma=arr.std(ddof=1),
                        p05=np.percentile(arr, 5),
                        p95=np.percentile(arr, 95))

# ❷  единые границы и бины
all_vals = np.concatenate(list(data.values()))
bins     = np.linspace(all_vals.min(), all_vals.max(), 70)

# ❸  график (накладываем гистограммы полупрозрачно)
plt.figure(figsize=(9, 4))
for label, arr in data.items():
    plt.hist(arr, bins=bins, alpha=0.45, edgecolor="k",
             density=True, label=label)

plt.xlabel("Final realized PnL")
plt.ylabel("Density")
plt.title(f"Deep-Hedge | {len(next(iter(data.values())))} runs per λ")
plt.legend()
plt.tight_layout()
plt.show()

# ❹  печать сводной таблицы
print("\n──────── summary by λ ────────")
for label, s in stats.items():
    print(f"{label:9} | μ={s['mu']:7.2f}  σ={s['sigma']:7.2f}  "
          f"5–95%=[{s['p05']:6.2f}; {s['p95']:6.2f}]")