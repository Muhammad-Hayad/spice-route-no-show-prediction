
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from datetime import date, time


# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.joblib"

FEATURE_COLUMNS = [
    "party_size",
    "day_of_week",
    "hour_of_day",
    "booking_channel_Phone",
    "booking_channel_Walk-in",
    "booking_channel_Website",
    "party_size_bin_Small",
    "party_size_bin_Medium",
    "party_size_bin_Large"
]

BOOKING_CHANNELS = ["Phone", "Walk-in", "Website"]


# ==============================
# LOAD SAVED MODEL
# ==============================

@st.cache_resource
def load_model():
    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]
    threshold = saved_data["threshold"]

    return model, threshold


# ==============================
# FEATURE BUILDING
# ==============================

def build_features(
    reservation_date,
    reservation_time,
    party_size,
    booking_channel
):

    # Validate party size
    if not isinstance(party_size, int) or not 1 <= party_size <= 8:
        raise ValueError("Party size must be between 1 and 8.")

    # Validate booking channel
    if booking_channel not in BOOKING_CHANNELS:
        raise ValueError("Unsupported booking channel.")

    # Extract date and time features
    day_of_week = reservation_date.weekday()
    hour_of_day = reservation_time.hour

    # Apply the same party-size bins used during training
    party_size_bin = pd.cut(
        pd.Series([party_size]),
        bins=[0, 2, 5, 8],
        labels=["Small", "Medium", "Large"]
    )

    # Create raw feature row
    input_data = pd.DataFrame({
        "party_size": [party_size],
        "day_of_week": [day_of_week],
        "hour_of_day": [hour_of_day],
        "booking_channel": [booking_channel],
        "party_size_bin": party_size_bin
    })

    # Apply the same one-hot encoding
    encoded_data = pd.get_dummies(
        input_data,
        columns=["booking_channel", "party_size_bin"],
        dtype=int
    )

    # Match the exact training feature columns and order
    encoded_data = encoded_data.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    return encoded_data


# ==============================
# STREAMLIT INTERFACE
# ==============================

st.set_page_config(
    page_title="Restaurant No-Show Predictor",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Restaurant Reservation No-Show Predictor")

st.write(
    "Predict the likelihood of a customer not showing up "
    "for their restaurant reservation."
)

st.divider()

# Reservation inputs
st.subheader("Reservation Details")

reservation_date = st.date_input(
    "Reservation Date",
    value=date.today()
)

reservation_time = st.time_input(
    "Reservation Time",
    value=time(18, 0)
)

party_size = st.number_input(
    "Party Size",
    min_value=1,
    max_value=8,
    value=2,
    step=1
)

booking_channel = st.selectbox(
    "Booking Channel",
    BOOKING_CHANNELS
)

st.divider()


# ==============================
# PREDICTION
# ==============================

if st.button("Predict No-Show", use_container_width=True):

    try:
        model, threshold = load_model()

        features = build_features(
            reservation_date,
            reservation_time,
            int(party_size),
            booking_channel
        )

        # Get probability for the positive class (no-show = 1)
        class_index = list(model.classes_).index(1)

        no_show_probability = model.predict_proba(
            features
        )[0][class_index]

        # Apply saved classification threshold
        prediction = int(no_show_probability >= threshold)

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "No-Show Probability",
            f"{no_show_probability:.2%}"
        )

        st.progress(float(no_show_probability))

        if prediction == 1:

            st.warning(
                "⚠️ Potential No-Show: This reservation "
                "has been flagged by the model."
            )

        else:

            st.success(
                "✅ Likely to Attend: The reservation "
                "has not been flagged by the model."
            )

        st.caption(
            f"Classification threshold: {threshold:.2f}"
        )

        st.info(
            "This prediction is an estimate, not a guarantee. "
            "Staff should use it as supporting information "
            "rather than automatically holding or releasing tables."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
