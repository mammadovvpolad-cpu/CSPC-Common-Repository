import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("chemicals_cancer.csv")

# Naive analysis: each chemical vs malignancy, using all 1000 patients
for chem in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(df[chem], df["malignancy"])
    rho, p_rho = stats.spearmanr(df[chem], df["malignancy"])
    print(f"{chem:8s} Pearson r={r:.3f} (p={p:.1e})  Spearman rho={rho:.3f}")

# Full correlation table, to see how everything relates
print(df.corr().round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, chem in zip(axes, ["benzene", "cadmium"]):
    ax.scatter(df[chem], df["malignancy"], alpha=0.4, s=12)
    ax.set_xlabel(chem)
    ax.set_ylabel("malignancy")
    ax.set_title(f"{chem} vs malignancy")
plt.tight_layout()
plt.savefig("naive.png", dpi=150)
plt.show()

# Part 7: hold pollution roughly constant
print("\n--- Restricted to similar pollution ---")
for lo, hi in [(40, 60), (45, 55)]:
    sub = df[(df["pollution_index"] > lo) & (df["pollution_index"] < hi)]
    print(f"\npollution between {lo} and {hi}: n={len(sub)}")
    for chem in ["benzene", "cadmium"]:
        r, p = stats.pearsonr(sub[chem], sub["malignancy"])
        print(f"  {chem:8s} r={r:.3f} (p={p:.1e})")

# Second check: partial correlation by regressing pollution out of both variables
import numpy as np
def residuals(y, x):
    slope, intercept, *_ = stats.linregress(x, y)
    return y - (intercept + slope * x)

print("\n--- Partial correlation (pollution removed) ---")
for chem in ["benzene", "cadmium"]:
    r, p = stats.pearsonr(residuals(df[chem], df["pollution_index"]),
                          residuals(df["malignancy"], df["pollution_index"]))
    print(f"  {chem:8s} r={r:.3f} (p={p:.1e})")

# Plot the restricted group
sub = df[(df["pollution_index"] > 40) & (df["pollution_index"] < 60)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, chem in zip(axes, ["benzene", "cadmium"]):
    ax.scatter(sub[chem], sub["malignancy"], alpha=0.5, s=14)
    ax.set_xlabel(chem)
    ax.set_ylabel("malignancy")
    ax.set_title(f"{chem} (pollution 40-60)")
plt.tight_layout()
plt.savefig("restricted.png", dpi=150)
plt.show()