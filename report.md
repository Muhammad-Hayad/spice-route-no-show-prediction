
# Restaurant Reservation No-Show Prediction

## 1. Project Overview

This project develops a machine learning model to predict whether a restaurant reservation will result in a no-show.

## 2. Feature Engineering

The following features were prepared:

- Day of the week
- Hour of the day
- Party size category
- Booking channel (one-hot encoded)

The dataset was divided into 80% training data and 20% testing data using a random state of 42.

## 3. Model Training

- Algorithm: Logistic Regression
- Parameters: Default parameters
- Classification threshold: 0.13
- Threshold selection: 5-fold stratified cross-validation using training data only.

## 4. Model Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 35.00% |
| Recall | 75.00% |
| F1 Score | 31.58% |

### Confusion Matrix

| Actual / Predicted | Showed | No-show |
|---|---:|---:|
| Showed | 8 | 24 |
| No-show | 2 | 6 |

![Confusion Matrix](confusion_matrix.png)

## 5. Baseline Comparison

The baseline model achieved 0% recall, while Logistic Regression achieved 75% recall using the selected classification threshold.

This demonstrates improved detection of no-show reservations, although the lower threshold also increases false-positive predictions.

## 6. Conclusion

The Logistic Regression model improves no-show detection compared with the baseline model. However, its accuracy and precision indicate that further improvements are needed.

The dataset contains only 200 synthetic reservations, so additional real-world data and further feature engineering are recommended before practical deployment.
