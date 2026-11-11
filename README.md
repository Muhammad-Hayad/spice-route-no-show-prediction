# Spice Route Restaurant – No-Show Prediction

## Project Overview
This project develops a baseline machine learning model to predict whether a restaurant reservation will result in a no-show.

The project uses a synthetic dataset of 200 reservations for demonstration purposes. It is not real customer data from Spice Route Restaurant.

## Dataset
The dataset contains the following features:

- `reservation_id`: Unique reservation identifier.
- `date`: Reservation date.
- `time`: Reservation time.
- `party_size`: Number of guests.
- `booking_channel`: Reservation booking method.
- `no_show`: Target variable (1 = No-show, 0 = Guest arrived).

## Technologies Used
- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Model
A DummyClassifier with the `most_frequent` strategy is used as the baseline model.

## Baseline Results
- Total reservations: 200
- Training records: 160
- Testing records: 40
- Test accuracy: 80%

Note: This accuracy is only a baseline benchmark. The DummyClassifier predicts the most frequent class and does not learn meaningful reservation patterns.

## Project Structure
```text
spice-route-no-show-prediction/
├── data/
│   ├── reservations.csv
│   ├── train.csv
│   └── test.csv
├── baseline.py
├── baseline_model.joblib
├── requirements.txt
├── README.md
└── LICENSE
```

## How to Run

1. Clone the repository.
2. Create and activate a Python 3.11 virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the baseline script:

```bash
python baseline.py
```

## Data Disclaimer
The dataset is synthetically generated and is intended only for educational and demonstration purposes.
