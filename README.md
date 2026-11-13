# Spice Route Restaurant – No-Show Prediction

## Live Demo

**Try the deployed application:** [Spice Route No-Show Predictor](https://spice-route-no-show-predictor.streamlit.app/)

## Project Overview

This project develops a machine learning application to estimate the probability that a restaurant reservation will result in a no-show.

It uses a synthetic dataset of 200 reservations for educational and demonstration purposes. The application allows users to enter reservation details and receive a no-show probability with a classification based on the model's selected threshold.

**Important:** This is a demonstration project, not a production-ready restaurant prediction system.

## Dataset

The dataset contains 200 synthetic reservation records with the following fields:

- `reservation_id`: Unique reservation identifier.
- `date`: Reservation date.
- `time`: Reservation time.
- `party_size`: Number of guests.
- `booking_channel`: Reservation booking method.
- `no_show`: Target variable (1 = No-show, 0 = Guest arrived).

The dataset is split into:
- Training records: 160
- Testing records: 40

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Model

The project initially used a `DummyClassifier` with the `most_frequent` strategy as a baseline.

The improved version uses **Logistic Regression** to estimate the probability of a reservation no-show.

### Feature Engineering

The model uses the following features:

- Party size
- Day of the week
- Hour of the day
- Booking channel (one-hot encoded)
- Party size category (Small, Medium, Large)

Feature engineering is performed consistently for training and application inputs. Test features are aligned with the training feature columns to maintain compatibility and avoid data leakage.

### Classification Threshold

The saved classification threshold is **0.13**.

A reservation is flagged as a potential no-show when its predicted probability is greater than or equal to this threshold.

## Model Evaluation and Limitations

The initial DummyClassifier achieved 80% test accuracy by predicting the most frequent class. This was only a baseline benchmark and did not identify meaningful reservation patterns.

The improved Logistic Regression model was evaluated on the 40-record test set. At the selected threshold of 0.13, it achieved approximately 35% accuracy.

The evaluation also showed 24 false positives and 8 true negatives. This means the model flags many reservations as potential no-shows even though the guests arrive.

The threshold prioritizes identifying potential no-shows at the cost of additional false alarms. In a real restaurant, these false positives could result in tables being unnecessarily held.

**Limitations:**
- The dataset is small and synthetically generated.
- Model performance is not sufficient for real-world operational decisions.
- Predictions are estimates, not guarantees.
- Further testing with real reservation data and additional model improvements are required.

## Streamlit Application

The application provides a simple interface where users can:

1. Select a reservation date.
2. Select a reservation time.
3. Enter a party size (1–8 guests).
4. Choose a booking channel (Phone, Walk-in, or Website).
5. View the predicted no-show probability.
6. See the classification based on the saved threshold.

## Project Structure

```text
spice-route-no-show-prediction/
├── data/
│   ├── reservations.csv
│   ├── features_train.csv
│   └── features_test.csv
├── feature_engineering.py
├── train_model.py
├── final_model.joblib
├── app.py
├── test_feature_builder.py
├── requirements.txt
├── README.md
├── report.md
└── .gitignore
```

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Muhammad-Hayad/spice-route-no-show-prediction.git
```

2. Navigate to the project directory:

```bash
cd spice-route-no-show-prediction
```

3. Create and activate a Python 3.11 virtual environment.

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Start the Streamlit application:

```bash
python -m streamlit run app.py
```

6. Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## GitHub Repository

[View Source Code on GitHub](https://github.com/Muhammad-Hayad/spice-route-no-show-prediction)

## Data Disclaimer

The dataset is synthetically generated and does not contain real customer information from Spice Route Restaurant. This project is intended for educational, experimental, and demonstration purposes only.