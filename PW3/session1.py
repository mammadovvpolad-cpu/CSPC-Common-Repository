import pandas as pd

df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
print(df.head())

cont = df[["age", "chol", "trestbps", "thalach"]]
print(cont.describe())

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), cont.columns):
    ax.hist(cont[col], bins=20, edgecolor="black")
    ax.axvline(cont[col].mean(), color="red", linestyle="--", label="mean")
    ax.axvline(cont[col].median(), color="green", linestyle=":", label="median")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
plt.savefig("distributions.png", dpi=150)
plt.show()

from scipy import stats

# Numerical checks
print("\nNormality checks")
for col in cont.columns:
    x = cont[col]
    W, p = stats.shapiro(x)
    print(f"{col:9s} skew={stats.skew(x):6.2f}  kurtosis={stats.kurtosis(x):6.2f}  Shapiro p={p:.4f}")

# Graphical check: Q-Q plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), cont.columns):
    stats.probplot(cont[col], dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot: {col}")
plt.tight_layout()
plt.savefig("qqplots.png", dpi=150)
plt.show()