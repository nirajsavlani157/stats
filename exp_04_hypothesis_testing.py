import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape)

alpha = 0.05
num_col, group_col = "Age", "Sex"
group_a, group_b = "male", "female"
cat_col1, cat_col2 = "Sex", "Survived"
anova_num_col, anova_group_col = "Fare", "Pclass"

def decide(p):
    return "Reject H0" if p < alpha else "Fail to reject H0"

print(df[group_col].value_counts())
print(df[anova_group_col].value_counts())

g1 = df.loc[df[group_col] == group_a, num_col].dropna()
g2 = df.loc[df[group_col] == group_b, num_col].dropna()
print(len(g1), g1.mean(), g1.std())
print(len(g2), g2.mean(), g2.std())

plt.figure(figsize=(6, 4))
sns.boxplot(data=df[df[group_col].isin([group_a, group_b])], x=group_col, y=num_col)
plt.title(f"Comparison of {num_col} by {group_col}")
plt.tight_layout()
plt.show()

t_stat, t_p = ttest_ind(g1, g2, equal_var=False)
print(t_stat, t_p, decide(t_p))

chi_df = df[[cat_col1, cat_col2]].dropna()
observed = pd.crosstab(chi_df[cat_col1], chi_df[cat_col2])

plt.figure(figsize=(6, 4))
sns.heatmap(observed, annot=True, fmt="d", cmap="Blues")
plt.title(f"Contingency Table ({cat_col1} vs {cat_col2})")
plt.tight_layout()
plt.show()

chi2, chi_p, dof, expected = chi2_contingency(observed)
print(chi2, dof, chi_p, decide(chi_p))

expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
print(expected_df.round(2))

groups_used = df[anova_group_col].value_counts().head(3).index.tolist()
anova_df = df[df[anova_group_col].isin(groups_used)]
samples = [anova_df.loc[anova_df[anova_group_col] == g, anova_num_col].dropna() for g in groups_used]

print(anova_df.groupby(anova_group_col)[anova_num_col].agg(["count", "mean"]))

plt.figure(figsize=(7, 4))
sns.boxplot(data=anova_df, x=anova_group_col, y=anova_num_col)
plt.title(f"{anova_num_col} across {anova_group_col}")
plt.tight_layout()
plt.show()

f_stat, anova_p = f_oneway(*samples)
print(f_stat, anova_p, decide(anova_p))

clean_anova = anova_df[[anova_num_col, anova_group_col]].dropna()
tukey = pairwise_tukeyhsd(clean_anova[anova_num_col], clean_anova[anova_group_col], alpha=alpha)
print(tukey)

results = pd.DataFrame({
    "Test": ["Welch's t-test", "Chi-square test", "One-way ANOVA"],
    "Statistic": [t_stat, chi2, f_stat],
    "p-value": [t_p, chi_p, anova_p]
})
results["Decision"] = results["p-value"].apply(decide)
print(results)
results.to_csv("hypothesis_testing_results.csv", index=False)
