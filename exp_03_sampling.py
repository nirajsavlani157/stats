import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape[0], df.shape[1])
print(df.select_dtypes(exclude=np.number).columns.tolist())
print(df.select_dtypes(include=np.number).columns.tolist())

num_col = "Fare"
stratify_col = "Pclass"
cluster_col = "Embarked"

N = len(df)
n = max(1, int(0.20 * N))
population_mean = df[num_col].mean()
print(N, n, population_mean)

srs_sample = df.sample(n=n, random_state=42)
print(len(srs_sample))
print(srs_sample.head())

print(df[stratify_col].value_counts())
sns.countplot(data=df, x=stratify_col)
plt.title("Distribution of Strata")
plt.show()

strat_sample = df.groupby(stratify_col, group_keys=False).sample(frac=n / N, random_state=42)
print(len(strat_sample))

strat_prop = pd.DataFrame({
    "Population %": df[stratify_col].value_counts(normalize=True) * 100,
    "Sample %": strat_sample[stratify_col].value_counts(normalize=True) * 100
}).round(2)
print(strat_prop)

clusters = df[cluster_col].dropna().unique()
k = max(1, int(0.30 * len(clusters)))
rng = np.random.default_rng(42)
selected = rng.choice(clusters, size=k, replace=False)
print(len(clusters), list(selected))

cluster_sample = df[df[cluster_col].isin(selected)].copy()
print(len(cluster_sample))

samples = {
    "Simple Random": srs_sample,
    "Stratified": strat_sample,
    "Cluster": cluster_sample
}

comparison = pd.DataFrame({
    "Sample Size": [len(s) for s in samples.values()],
    "Sample Mean": [s[num_col].mean() for s in samples.values()]
}, index=samples.keys())

comparison["Difference from Population Mean"] = comparison["Sample Mean"] - population_mean
print(comparison.round(3))

plt.figure(figsize=(8, 5))
comparison["Sample Mean"].plot(kind="bar", rot=0)
plt.axhline(population_mean, color="red", linestyle="--")
plt.ylabel(f"Mean of {num_col}")
plt.title("Sample Means vs Population Mean")
plt.tight_layout()
plt.show()

srs_sample.to_csv("SRS_sample.csv", index=False)
strat_sample.to_csv("Stratified_sample.csv", index=False)
cluster_sample.to_csv("Cluster_sample.csv", index=False)
