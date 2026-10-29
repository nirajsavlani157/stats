import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(len(X_train), len(X_test))

results, preds, probs = [], {}, {}

def evaluate(name, model):
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)
    probs[name] = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
        "Accuracy": accuracy_score(y_test, preds[name]),
        "Precision": precision_score(y_test, preds[name], zero_division=0),
        "Recall": recall_score(y_test, preds[name], zero_division=0),
        "F1": f1_score(y_test, preds[name], zero_division=0),
        "AUC": roc_auc_score(y_test, probs[name])
    })
    print(name, results[-1])
    return model

dt = evaluate("Decision Tree (full)", DecisionTreeClassifier(criterion="gini", random_state=42))
print(dt.get_depth(), dt.get_n_leaves())

dt_small = evaluate("Decision Tree (restricted)",
                    DecisionTreeClassifier(criterion="gini", max_depth=4,
                                           min_samples_split=10, min_samples_leaf=5, random_state=42))
print(dt_small.get_depth(), dt_small.get_n_leaves())

plt.figure(figsize=(18, 8))
plot_tree(dt_small, max_depth=3, feature_names=predictors,
          class_names=["Perished", "Survived"],
          filled=True, rounded=True, fontsize=9)
plt.tight_layout()
plt.show()

rf = evaluate("Random Forest",
              RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1))
print(round(rf.oob_score_, 4))

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
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, scoring="accuracy", random_state=42)
importance = pd.DataFrame({
    "Decision Tree": dt.feature_importances_,
    "Random Forest": rf.feature_importances_,
    "Permutation (mean)": perm.importances_mean,
    "Permutation (std)": perm.importances_std
}, index=predictors).sort_values("Random Forest", ascending=False)
print(importance.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
importance[["Decision Tree", "Random Forest"]].plot(kind="barh", ax=axes[0])
axes[0].invert_yaxis()
axes[1].barh(importance.index, importance["Permutation (mean)"], xerr=importance["Permutation (std)"], color="coral")
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()

tree_rows = []
for n in [10, 25, 50, 100, 200]:
    cv = cross_validate(RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1),
                        X_train, y_train, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    tree_rows.append({
        "Trees": n,
        "CV Accuracy": cv["test_accuracy"].mean(),
        "CV F1": cv["test_f1"].mean(),
        "CV AUC": cv["test_roc_auc"].mean()
    })
tree_results = pd.DataFrame(tree_rows)
print(tree_results.round(4))

plt.figure(figsize=(8, 4.5))
plt.plot(tree_results["Trees"], tree_results["CV Accuracy"], marker="o", label="CV Accuracy")
plt.plot(tree_results["Trees"], tree_results["CV AUC"], marker="s", label="CV AUC")
plt.xlabel("Number of Trees")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

comparison = pd.DataFrame(results).set_index("Model")
print(comparison.round(4))

comparison.drop(columns="Train Accuracy").plot(kind="bar", rot=0, figsize=(11, 4.5), ylim=(0, 1))
plt.tight_layout()
plt.show()

comparison.to_csv("Experiment_9_Model_Results.csv")
importance.to_csv("Experiment_9_Feature_Importance.csv")
tree_results.to_csv("Experiment_9_Tree_Count.csv", index=False)
