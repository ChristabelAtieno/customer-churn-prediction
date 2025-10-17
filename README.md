# Telecom Churn Prediction

This project predicts customer churn for a telecom company using **Logistic Regression** and **XGBoost**, tracks experiments with **MLflow**, and containerizes the application with **Docker**.

---

## Features

- **Data Preprocessing:** Categorical encoding, and feature scaling.
- **Feature Engineering:** Creates relevant features for churn prediction.
- **Models:**
  - Logistic Regression (baseline)
  - XGBoost (advanced)
- **Experiment Tracking:** Uses MLflow to log:
  - Metrics (accuracy, recall, precision, F1)
  - Parameters
  - Models
- **Cross-validation:** Evaluates model performance using Stratified K-Folds.
- **Containerization:** Dockerfile to run the application and MLflow tracking server in a container.

