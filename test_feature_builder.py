
import joblib
import pandas as pd

MODEL_PATH = "final_model.joblib"
TRAIN_PATH = "data/features_train.csv"

# Load saved model and expected feature order
saved = joblib.load(MODEL_PATH)
model = saved["model"]

feature_columns = list(model.feature_names_in_)


def build_features(reservation_date, reservation_time,
                   party_size, booking_channel):

    df = pd.DataFrame([{
        "date": reservation_date,
        "time": reservation_time,
        "party_size": party_size,
        "booking_channel": booking_channel
    }])

    # Date and time transformations
    df["date"] = pd.to_datetime(df["date"])
    df["time"] = pd.to_datetime(
        df["time"], format="%H:%M"
    )

    df["day_of_week"] = df["date"].dt.day_of_week
    df["hour_of_day"] = df["time"].dt.hour

    # Party size categorization
    df["party_size_bin"] = pd.cut(
        df["party_size"],
        bins=[0, 2, 5, 8],
        labels=["Small", "Medium", "Large"]
    )

    # One-hot encoding using the original training approach
    df = df.drop(columns=["date", "time"])

    df = pd.get_dummies(
        df,
        columns=["booking_channel", "party_size_bin"],
        dtype=int
    )

    # Enforce exact training feature names and order
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df


# Load one actual training row
train_data = pd.read_csv(TRAIN_PATH)
training_row = train_data.drop(columns=["no_show"]).iloc[0]

# Reconstruct its raw inputs
channel_columns = [
    "booking_channel_Phone",
    "booking_channel_Walk-in",
    "booking_channel_Website"
]

channel = next(
    col.replace("booking_channel_", "")
    for col in channel_columns
    if training_row[col] == 1
)

weekday = int(training_row["day_of_week"])

# 2024-01-01 was a Monday
reservation_date = (
    pd.Timestamp("2024-01-01") +
    pd.Timedelta(days=weekday)
).strftime("%Y-%m-%d")

reservation_time = f"{int(training_row['hour_of_day']):02d}:00"

# Build features from reconstructed raw inputs
app_row = build_features(
    reservation_date,
    reservation_time,
    int(training_row["party_size"]),
    channel
).iloc[0]

# Compare training row with app-generated row
comparison = pd.DataFrame({
    "Training Row": training_row,
    "App Generated Row": app_row
})

print("\n========== FEATURE COMPARISON ==========")
print(comparison)

print("\nFeature order matches:",
      list(training_row.index) == list(app_row.index))

print("All nine values match:",
      training_row.equals(app_row))

print("========================================")
