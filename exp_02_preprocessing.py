import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape[0], df.shape[1])
df.info()

missing = pd.DataFrame({
    "Missing": df.isnull().sum(),
    "Percent": (df.isnull().mean() * 100).round(2)
})
print(missing[missing["Missing"] > 0])

df_clean = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore").copy()
numeric_cols = df_clean.select_dtypes(include=np.number).columns
categorical_cols = df_clean.select_dtypes(exclude=np.number).columns

for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print(df_clean.isnull().sum().sum())

for col in categorical_cols:
    print(col, df_clean[col].unique())

df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(df_encoded.shape)
print(df_encoded.head())

demo_col = "Sex"
codes = LabelEncoder().fit_transform(df_clean[demo_col].astype(str))
print(pd.DataFrame({demo_col: df_clean[demo_col], "Encoded": codes}).head())

std_df = pd.DataFrame(StandardScaler().fit_transform(df_clean[numeric_cols]), columns=numeric_cols)
mm_df = pd.DataFrame(MinMaxScaler().fit_transform(df_clean[numeric_cols]), columns=numeric_cols)

print(std_df.agg(["mean", "std"]).round(2))
print(mm_df.agg(["min", "max"]).round(2))

plt.figure(figsize=(10, 6))
sns.boxplot(data=std_df)
plt.title("Box Plot of Numerical Variables")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

rows = []
for col in numeric_cols:
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
    rows.append([col, q1, q3, iqr, lo, hi, n])

iqr_table = pd.DataFrame(rows, columns=["Column", "Q1", "Q3", "IQR", "Lower", "Upper", "Outliers"])
print(iqr_table)

outlier_col = "Fare"
lo, hi = iqr_table.set_index("Column").loc[outlier_col, ["Lower", "Upper"]]
df_final = df_clean[df_clean[outlier_col].between(lo, hi)].copy()
print(len(df_clean), len(df_final))

df_final.to_csv("preprocessed_dataset.csv", index=False)
