
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "reservations.csv"

# Load original dataset
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["reservation_id"])

print("Original dataset loaded!")
print("Original shape:", df.shape)

# Date and time feature extraction
df["date"] = pd.to_datetime(df["date"])
df["time"] = pd.to_datetime(
    df["time"],
    format="%H:%M"
)

df["day_of_week"] = df["date"].dt.day_of_week
df["hour_of_day"] = df["time"].dt.hour

# Party size feature engineering
df["party_size_bin"] = pd.cut(
    df["party_size"],
    bins=[0, 2, 5, 8],
    labels=["Small", "Medium", "Large"]
)

# Remove original date and time columns
df = df.drop(columns=["date", "time"])

print("\nFeature engineering completed!")
print("Engineered dataset shape:", df.shape)

# Separate features and target BEFORE encoding
X = df.drop(columns=["no_show"])
y = df["no_show"]

# Split original data BEFORE encoding
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Encode training data only
X_train = pd.get_dummies(
    X_train,
    columns=["booking_channel", "party_size_bin"],
    dtype=int
)

# Align test columns with training columns
X_test = pd.get_dummies(
    X_test,
    columns=["booking_channel", "party_size_bin"],
    dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Combine features and target
train_data = X_train.copy()
train_data["no_show"] = y_train

test_data = X_test.copy()
test_data["no_show"] = y_test

# Save datasets
train_data.to_csv(
    DATA_DIR / "features_train.csv",
    index=False
)

test_data.to_csv(
    DATA_DIR / "features_test.csv",
    index=False
)

print("\nTrain/test split completed!")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining features:", X_train.columns.tolist())
print("Testing features:", X_test.columns.tolist())

print("\nTrain columns match test columns:",
      X_train.columns.equals(X_test.columns))

print("\nFeature datasets saved successfully!")
print("\nTASK 2 FEATURE ENGINEERING COMPLETED!")
