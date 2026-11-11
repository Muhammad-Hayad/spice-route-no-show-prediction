
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score



DATA_DIR = Path("data")
MODEL_DIR = Path("models")

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "reservations.csv"
MODEL_PATH = Path("baseline_model.joblib")


np.random.seed(42)

number_of_reservations = 200

dates = pd.date_range(
    start="2026-01-01",
    periods=90,
    freq="D"
)

data = pd.DataFrame({
    "reservation_id": range(1, number_of_reservations + 1),

    "date": np.random.choice(
        dates.strftime("%Y-%m-%d"),
        number_of_reservations
    ),

    "time": np.random.choice(
        ["12:00", "13:00", "14:00", "18:00",
         "19:00", "20:00", "21:00"],
        number_of_reservations
    ),

    "party_size": np.random.randint(
        1, 9, number_of_reservations
    ),

    "booking_channel": np.random.choice(
        ["Website", "Phone", "Walk-in"],
        number_of_reservations,
        p=[0.5, 0.3, 0.2]
    )
})


no_show_probability = (
    0.12
    + (data["party_size"] * 0.015)
    + (data["booking_channel"] == "Website") * 0.04
)

no_show_probability = no_show_probability.clip(0.05, 0.60)

data["no_show"] = (
    np.random.random(number_of_reservations)
    < no_show_probability
).astype(int)



data.to_csv(DATA_PATH, index=False)

print("Dataset generated successfully!")
print("Total reservations:", len(data))



df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Original shape:", df.shape)



df = df.drop_duplicates()

df = df.dropna()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"])

df["time"] = pd.to_datetime(
    df["time"],
    format="%H:%M",
    errors="coerce"
).dt.strftime("%H:%M")

df = df.dropna(subset=["time"])

df = df.drop(columns=["reservation_id"])

print("Cleaned dataset shape:", df.shape)


X = df.drop(columns=["no_show"])
y = df["no_show"]

X = pd.get_dummies(
    X,
    columns=["date", "time", "booking_channel"],
    drop_first=True
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/test split completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))



train_data = X_train.copy()
train_data["no_show"] = y_train

test_data = X_test.copy()
test_data["no_show"] = y_test

train_data.to_csv(DATA_DIR / "train.csv", index=False)
test_data.to_csv(DATA_DIR / "test.csv", index=False)

print("\nTraining and testing CSV files saved!")



model = DummyClassifier(
    strategy="most_frequent"
)

model.fit(X_train, y_train)

print("\nDummyClassifier trained successfully!")


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== BASELINE RESULTS ==========")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("======================================")


joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model file:", MODEL_PATH)

print("\nTASK 1 BASELINE COMPLETED!")
