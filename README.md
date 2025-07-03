# Heart-rate-predictor
# Health Monitoring System using Machine Learning

A robust machine learning project for heart rate prediction using physiological sensor data from Empatica E4 wearables. This project implements both traditional machine learning and deep learning (CNN + LSTM) models to deliver accurate, real-time heart rate monitoring.

---

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement & Objectives](#problem-statement--objectives)
- [Python Packages Used](#python-packages-used)
- [Source Code Overview](#source-code-overview)
- [Implementation Results](#implementation-results)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

Heart rate prediction is essential for health monitoring, particularly for patients with cardiovascular diseases. With wearable sensors like Empatica E4, real-time heart rate data and other vital signals (BVP, ACC_x, ACC_y, ACC_z) can be captured.  
This project focuses on predicting heart rate from such sensor data using various machine learning and deep learning techniques.

**Project workflow:**
1. Data Preprocessing: Clean and merge sensor data based on timestamps.
2. Feature Extraction and Normalization: Extract features and normalize the data.
3. Modeling: Implement and evaluate Linear Regression, Random Forest, XGBoost, SVR, and a hybrid CNN + LSTM model.
4. Evaluation: Compare models using MAE, RMSE, R², and Pearson correlation.
5. Statistical Comparison: Perform statistical tests to compare model performance.

---

## Problem Statement & Objectives

**Problem Statement:**  
Predict heart rate (HR) based on data from multiple sensors such as BVP (Blood Volume Pulse), accelerometer (ACC), and HR data. The goal is to develop a robust system that can accurately predict heart rate in real-time.

**Objectives:**
- **Data Preprocessing and Feature Extraction:**  
  - Load, clean, and merge the sensor data.
  - Extract relevant features (BVP, ACC).
- **Train Multiple Models:**  
  - Train Linear Regression, Random Forest, SVR, XGBoost, and CNN + LSTM models for heart rate prediction.
- **Model Evaluation:**  
  - Compare model performance using various evaluation metrics.
- **Statistical Analysis:**  
  - Perform statistical tests to validate model comparisons.

---

## Python Packages Used

| Package      | Functions Used                                 | Purpose                                       |
|--------------|------------------------------------------------|-----------------------------------------------|
| pandas       | read_csv, merge_asof                           | Data loading, manipulation, merging           |
| numpy        | array, mean, sqrt                              | Numerical operations, arrays                  |
| matplotlib   | plot, bar, show                                | Visualization                                 |
| seaborn      | barplot, heatmap                               | Statistical plotting                          |
| scikit-learn | StandardScaler, train_test_split, mean_absolute_error, mean_squared_error, LinearRegression, RandomForestRegressor, SVR | Preprocessing, ML algorithms, evaluation      |
| xgboost      | XGBRegressor                                   | Gradient boosting regression                  |
| tensorflow   | Sequential, LSTM, Conv1D, MaxPooling1D, Dropout, ModelCheckpoint | Deep learning (CNN + LSTM)                    |
| scipy        | ttest_rel, pearsonr                            | Statistical analysis, correlation             |

---

## Source Code Overview

Below is a high-level summary of the implementation steps.  
(For full code, see your project files.)

**1. Data Extraction and Loading**
- Upload and extract Empatica E4 data (`S3_E4.zip`).
- Load BVP, ACC, and HR signals with timestamps.

**2. Data Merging**
- Merge BVP, ACC, and HR data using nearest timestamps.

**3. Feature Engineering**
- Select features: `BVP`, `ACC_x`, `ACC_y`, `ACC_z`.
- Normalize features using `StandardScaler`.
- Create sliding window sequences (window size = 128, stride = 64).

**4. Train/Test Split**
- 80% for training, 20% for testing.
- Flatten data for traditional ML models.

**5. Model Training**
- Train the following models:
  - Linear Regression
  - Random Forest (`n_estimators=100`)
  - Support Vector Regressor
  - XGBoost (`n_estimators=100, learning_rate=0.1`)
- For deep learning:
    ```
    Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(window_size, n_features)),
        MaxPooling1D(2),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ```
- Compile with Adam optimizer, MSE loss, and MAE metric.

**6. Evaluation**
- Metrics: MAE, RMSE, R², Pearson correlation.
- Bar charts for visual comparison.
- Paired t-tests for statistical validation.

---

## Implementation Results

- All models were evaluated on MAE, RMSE, R² Score, and Pearson Correlation.
- The CNN + LSTM model achieved the best performance among all models.
- Visualizations (bar charts) provided clear model comparison.
- Statistical tests confirmed the significance of deep learning model improvements.

---

## Conclusion

This project demonstrates the effective use of sensor data for heart rate prediction using both traditional and deep learning models.  
The CNN + LSTM hybrid model outperformed the machine learning models, highlighting the value of deep learning for time-series predictions.  
**Future work:** Optimize model hyperparameters and test on larger datasets for further improvements.

---

## References

1. IEEE DataPort: PPG + DALIA dataset for real-time health monitoring analysis.


