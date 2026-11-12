
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Project paths
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "features_train.csv"
TEST_PATH = DATA_DIR / "features_test.csv"

MODEL_PATH = Path("final_model.joblib")
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")


# Load corrected feature datasets
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

print("Feature datasets loaded successfully!")

X_train = train_data.drop(columns=["no_show"])
y_train = train_data["no_show"]

X_test = test_data.drop(columns=["no_show"])
y_test = test_data["no_show"]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Cross-validation setup (training data only)
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Generate out-of-fold training probabilities
cv_model = LogisticRegression(max_iter=1000)

cv_probabilities = cross_val_predict(
    cv_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]


# Compare candidate thresholds
thresholds = np.arange(0.05, 0.51, 0.01)

threshold_results = []

for threshold in thresholds:

    cv_predictions = (
        cv_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        cv_predictions,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_train, cv_predictions),
        "Precision": precision_score(
            y_train, cv_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_train, cv_predictions, zero_division=0
        ),
        "F1": f1_score(
            y_train, cv_predictions, zero_division=0
        ),
        "FP": fp,
        "FN": fn
    })


threshold_df = pd.DataFrame(threshold_results)

print("\n========== TRAINING-ONLY CV THRESHOLD COMPARISON ==========")
print(
    threshold_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# Select threshold using cross-validation F1.
# If F1 is tied, prefer the higher threshold to reduce false positives.
best_row = threshold_df.sort_values(
    by=["F1", "Threshold"],
    ascending=[False, False]
).iloc[0]

best_threshold = best_row["Threshold"]

print("\n========== THRESHOLD SELECTION ==========")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"CV Accuracy: {best_row['Accuracy']:.4f}")
print(f"CV Precision: {best_row['Precision']:.4f}")
print(f"CV Recall: {best_row['Recall']:.4f}")
print(f"CV F1: {best_row['F1']:.4f}")
print(f"CV False Positives: {best_row['FP']}")
print(f"CV False Negatives: {best_row['FN']}")
print("=========================================")


# Train final Logistic Regression on all training records
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Final evaluation on untouched test data
test_probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    test_probabilities >= best_threshold
).astype(int)


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

lr_cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = lr_cm.ravel()


print("\n========== LOGISTIC REGRESSION RESULTS ==========")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)")

print("\nConfusion Matrix:")
print(lr_cm)

print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
print("=================================================")


# Baseline comparison
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

baseline_recall = recall_score(
    y_test, baseline_predictions, zero_division=0
)

baseline_f1 = f1_score(
    y_test, baseline_predictions, zero_division=0
)

baseline_cm = confusion_matrix(
    y_test,
    baseline_predictions,
    labels=[0, 1]
)

print("\n========== BASELINE COMPARISON ==========")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline Recall:   {baseline_recall:.4f}")
print(f"Baseline F1:       {baseline_f1:.4f}")
print("\nBaseline Confusion Matrix:")
print(baseline_cm)

print("\nLogistic Regression vs Baseline:")
print(f"Accuracy: {accuracy:.4f} vs {baseline_accuracy:.4f}")
print(f"Recall:   {recall:.4f} vs {baseline_recall:.4f}")
print(f"F1:       {f1:.4f} vs {baseline_f1:.4f}")
print("=========================================")


# Save confusion matrix visualization
ConfusionMatrixDisplay(
    confusion_matrix=lr_cm,
    display_labels=["Showed", "No-show"]
).plot(cmap="Blues")

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
plt.close()

print("\nConfusion matrix saved:", CONFUSION_MATRIX_PATH)


# Save final model and selected threshold
joblib.dump(
    {
        "model": model,
        "threshold": best_threshold
    },
    MODEL_PATH
)

print("\nModel and threshold saved successfully!")
print("Model file:", MODEL_PATH)

print("\nTASK 2 MODEL TRAINING COMPLETED!")
