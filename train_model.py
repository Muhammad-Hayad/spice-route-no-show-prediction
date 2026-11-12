
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


# Project paths
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "features_train.csv"
TEST_PATH = DATA_DIR / "features_test.csv"

MODEL_PATH = Path("final_model.joblib")
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")


train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

print("Feature datasets loaded successfully!")

X_train = train_data.drop(columns=["no_show"])
y_train = train_data["no_show"]

X_test = test_data.drop(columns=["no_show"])
y_test = test_data["no_show"]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


model = LogisticRegression()
model.fit(X_train, y_train)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_probabilities = cross_val_predict(
    LogisticRegression(),
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

thresholds = np.arange(0.05, 0.51, 0.01)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    cv_predictions = (cv_probabilities >= threshold).astype(int)

    current_f1 = f1_score(
        y_train,
        cv_predictions,
        zero_division=0
    )

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print("\n========== THRESHOLD SELECTION ==========")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Cross-validation F1: {best_f1:.4f}")
print("=========================================")


test_probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)


print("\n========== LOGISTIC REGRESSION RESULTS ==========")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Recall:   {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score: {f1:.4f} ({f1 * 100:.2f}%)")
print("=================================================")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1],
    display_labels=["Showed", "No-show"],
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
plt.close()

print("\nConfusion matrix saved:", CONFUSION_MATRIX_PATH)


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
