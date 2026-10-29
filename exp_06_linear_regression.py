import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape)

target = "Fare"
predictors = ["Pclass", "Age", "SibSp", "Parch"]
reg_df = df[[target] + predictors].dropna().copy()
print(len(reg_df))
print(reg_df.describe().T)

plt.figure(figsize=(7, 5))
sns.heatmap(reg_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Regression Variables")
plt.tight_layout()
plt.show()

X, y = reg_df[predictors], reg_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(len(X_train), len(X_test))

model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(model.summary())

coefficients = pd.DataFrame({
    "Coefficient": model.params,
    "Std Error": model.bse,
    "t-statistic": model.tvalues,
    "p-value": model.pvalues
})
print(coefficients.round(4))

y_pred = model.predict(sm.add_constant(X_test))
mse = mean_squared_error(y_test, y_pred)
rmse, mae = np.sqrt(mse), mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(model.rsquared, model.rsquared_adj)
print(test_r2, rmse, mae)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.show()

X_vif = sm.add_constant(X_train)
vif = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif = vif[vif["Variable"] != "const"]
print(vif.round(3))

plt.figure(figsize=(7, 4))
sns.barplot(data=vif, x="Variable", y="VIF")
plt.axhline(5, color="red", linestyle="--")
plt.title("Variance Inflation Factor")
plt.tight_layout()
plt.show()

fitted, residuals = model.fittedvalues, model.resid
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.scatterplot(x=fitted, y=residuals, ax=axes[0])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set(xlabel="Fitted values", ylabel="Residuals")

sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
axes[1].set(xlabel="Residual")

sm.qqplot(residuals, line="45", fit=True, ax=axes[2])
plt.tight_layout()
plt.show()

influence = OLSInfluence(model)
diagnostics = pd.DataFrame({
    "Actual": y_train,
    "Fitted": fitted,
    "Residual": residuals,
    "Std. Residual": influence.resid_studentized_internal,
    "Cook's D": influence.cooks_distance[0]
})
n = len(X_train)
threshold = 4 / n

plt.figure(figsize=(10, 4.5))
plt.stem(np.arange(n), diagnostics["Cook's D"], markerfmt=",", basefmt=" ")
plt.axhline(threshold, color="red", linestyle="--")
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.tight_layout()
plt.show()

influential = diagnostics[diagnostics["Cook's D"] > threshold]
print(len(influential))

summary = pd.DataFrame({
    "Metric": ["Training R2", "Adjusted R2", "Test R2", "Test MSE", "Test RMSE", "Test MAE", "Maximum VIF", "Influential observations"],
    "Value": [model.rsquared, model.rsquared_adj, test_r2, mse, rmse, mae, vif["VIF"].max(), len(influential)]
})
print(summary)

coefficients.to_csv("regression_coefficients.csv")
vif.to_csv("vif_results.csv", index=False)
diagnostics.to_csv("regression_diagnostics.csv")
summary.to_csv("regression_summary.csv", index=False)
