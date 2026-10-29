import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

print(df.shape[0], df.shape[1])
print(df.columns.tolist())
df.info()
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=np.number).drop(columns=["PassengerId"], errors="ignore")
print(numeric_df.describe().T)

numeric_df.hist(figsize=(12, 8), bins=15, edgecolor="black")
plt.suptitle("Histograms of Numerical Variables")
plt.tight_layout()
plt.show()

corr = numeric_df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
pairs = corr.where(mask).stack()
print(pairs.sort_values(key=abs, ascending=False).head(10))
