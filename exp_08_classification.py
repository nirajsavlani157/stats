import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)

df = pd.read_csv("Titanic-Dataset.csv")
print(df.shape)

target = "Survived"
positive_class = 1
predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
model_df = df[[target] + predictors].dropna()

X = model_df[predictors]
y = (model_df[target] == positive_class).astype(int)
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(len(X_train), len(X_test))

results, preds, probs = [], {}, {}

def evaluate(name, classifier):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)
    probs[name] = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds[name]),
        "Precision": precision_score(y_test, preds[name], zero_division=0),
        "Recall": recall_score(y_test, preds[name], zero_division=0),
        "F1": f1_score(y_test, preds[name], zero_division=0),
        "AUC": roc_auc_score(y_test, probs[name])
    })
    print(name, results[-1])
    return model

logistic = evaluate("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42))
coef = logistic.named_steps["classifier"].coef_[0]
coef_table = pd.DataFrame({"Coefficient": coef, "Odds Ratio": np.exp(coef)}, index=predictors)
print(coef_table.round(4))

knn = evaluate("KNN (K=5)", KNeighborsClassifier(n_neighbors=5))
nb = evaluate("Gaussian Naive Bayes", GaussianNB())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pred) in zip(axes, preds.items()):
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(title=name, xlabel="Predicted", ylabel="Actual")
plt.tight_layout()
plt.show()

for name, pred in preds.items():
    print(name)
    print(classification_report(y_test, pred, zero_division=0))

plt.figure(figsize=(8, 6))
for name, prob in probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

k_rows = []
for k in [1, 3, 5, 7, 9, 11, 15]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    cv = cross_validate(pipe, X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    k_rows.append({
        "K": k,
        "CV Accuracy": cv["test_accuracy"].mean(),
        "CV F1": cv["test_f1"].mean(),
        "CV AUC": cv["test_roc_auc"].mean()
    })
knn_results = pd.DataFrame(k_rows)
print(knn_results.round(4))

plt.figure(figsize=(8, 4.5))
plt.plot(knn_results["K"], knn_results["CV Accuracy"], marker="o", label="CV Accuracy")
plt.plot(knn_results["K"], knn_results["CV AUC"], marker="s", label="CV AUC")
plt.xlabel("K")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

comparison = pd.DataFrame(results).set_index("Model")
print(comparison.round(4))

comparison.plot(kind="bar", rot=0, figsize=(11, 4.5), ylim=(0, 1))
plt.tight_layout()
plt.show()

comparison.to_csv("Experiment_8_Classification_Results.csv")
coef_table.to_csv("Experiment_8_Logistic_Coefficients.csv")
knn_results.to_csv("Experiment_8_KNN_Results.csv", index=False)
