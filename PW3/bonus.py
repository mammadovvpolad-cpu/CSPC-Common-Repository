import numpy as np
import pandas as pd

df = pd.read_csv("heart.csv")

p = df["target"].value_counts(normalize=True)
print(p)

H = -(p * np.log2(p)).sum()
print(f"Entropy of target: {H:.3f} bits (max for 2 classes = 1 bit)")