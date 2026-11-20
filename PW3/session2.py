import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")
dis = df[df["target"] == 1]["thalach"]
hea = df[df["target"] == 0]["thalach"]

for name, g in [("disease", dis), ("healthy", hea)]:
    sem = stats.sem(g)
    ci = stats.t.ppf(0.975, len(g) - 1) * sem
    print(f"{name:8s} n={len(g)} mean={g.mean():.1f} std={g.std():.1f} SEM={sem:.2f} 95% CI = +/- {ci:.2f}")

# Welch t-test (no equal-variance assumption) and Mann-Whitney (no normality assumption)
t, p_t = stats.ttest_ind(dis, hea, equal_var=False)
U, p_u = stats.mannwhitneyu(dis, hea, alternative="two-sided")
print(f"Welch t-test: t={t:.2f}, p={p_t:.2e}")
print(f"Mann-Whitney: U={U:.0f}, p={p_u:.2e}")

# Plot the two means with 95% confidence intervals
means = [dis.mean(), hea.mean()]
cis = [stats.t.ppf(0.975, len(g) - 1) * stats.sem(g) for g in (dis, hea)]
plt.errorbar(["disease", "healthy"], means, yerr=cis, fmt="o", capsize=8, markersize=8)
plt.ylabel("Max heart rate, thalach (bpm)")
plt.title("Mean thalach with 95% confidence interval")
plt.savefig("thalach_means.png", dpi=150)
plt.show()

r, p_r = stats.pearsonr(df["age"], df["thalach"])
rho, p_rho = stats.spearmanr(df["age"], df["thalach"])
print(f"Pearson r={r:.3f} (p={p_r:.1e}), Spearman rho={rho:.3f} (p={p_rho:.1e})")

slope, intercept, *_ = stats.linregress(df["age"], df["thalach"])
plt.figure()
plt.scatter(df["age"], df["thalach"], alpha=0.6)
xs = np.array([df["age"].min(), df["age"].max()])
plt.plot(xs, intercept + slope * xs, color="red", label=f"slope = {slope:.2f} bpm/year")
plt.xlabel("Age (years)")
plt.ylabel("Max heart rate, thalach (bpm)")
plt.title(f"Age vs thalach (r = {r:.2f})")
plt.legend()
plt.savefig("age_thalach.png", dpi=150)
plt.show()