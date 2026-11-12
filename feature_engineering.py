
import pandas as pd
from pathlib import Path

# Project directories
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "reservations.csv"

# Load original dataset
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["reservation_id"])

print("Original dataset loaded!")
print("Original shape:", df.shape)

# Convert date and time columns
df["date"] = pd.to_datetime(df["date"])
df["time"] = pd.to_datetime(
    df["time"],
    format="%H:%M"
)

# Feature 1: Day of the week
df["day_of_week"] = df["date"].dt.day_of_week

# Feature 2: Hour of the day
df["hour_of_day"] = df["time"].dt.hour

# Feature 3: Party size categories
df["party_size_bin"] = pd.cut(
    df["party_size"],
    bins=[0, 2, 5, 8],
    labels=["Small", "Medium", "Large"]
)

# Remove original date and time columns
df = df.drop(columns=["date", "time"])

# One-hot encode booking channel and party size category
df = pd.get_dummies(
    df,
    columns=["booking_channel", "party_size_bin"],
    dtype=int
)

print("\nFeature engineering completed!")
print("Engineered dataset shape:", df.shape)

print("\nEngineered columns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nTASK 2 FEATURE ENGINEERING COMPLETED!")



from sklearn.model_selection import train_test_split

# Split features and target
X = df.drop(columns=["no_show"])
y = df["no_show"]

# Train/test split (same settings as Task 1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine features and target
train_data = X_train.copy()
train_data["no_show"] = y_train

test_data = X_test.copy()
test_data["no_show"] = y_test

# Save datasets
train_data.to_csv(DATA_DIR / "features_train.csv", index=False)
test_data.to_csv(DATA_DIR / "features_test.csv", index=False)

print("\nTrain/test split completed!")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nFeature datasets saved successfully!")

