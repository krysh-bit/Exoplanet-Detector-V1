# Exoplanet Detector V1

A machine learning project that classifies Kepler Objects of Interest (KOIs) into three categories: **CONFIRMED**, **CANDIDATE**, and **FALSE POSITIVE**.

## 1. Project Overview

The goal of this project is to build a supervised machine learning classification model that predicts the disposition of potential exoplanet candidates using data from NASA's Kepler mission.

This project explores how planetary transit measurements and host-star properties can be used to distinguish between confirmed planets, candidates, and false positives.

## 2. Dataset

- **Source:** Kaggle – Kepler Exoplanet Search Results
- **Dataset size:** Approximately 9,564 observations
- **Target variable:** `koi_disposition`
- **Classification categories:**
  - CONFIRMED
  - CANDIDATE
  - FALSE POSITIVE

## 3. Exploratory Data Analysis

The initial data exploration focused on understanding the dataset, identifying patterns, and examining the distribution of important features.

The analysis included:

- Target class distribution.
- Feature distributions using histograms.
- Comparison of selected planetary measurements across classification categories.
- Identification of missing values and extreme values.

Some observations from the analysis:

- False positives showed larger median planetary radius values than confirmed planets and candidates.
- Transit depth and impact parameter distributions varied across the classification categories.
- Several numerical features contained missing values and skewed distributions.

These observations helped guide feature selection and preprocessing.

## 4. Data Preprocessing

The following preprocessing steps were performed:

- Selected eight numerical features for model training.
- Separated the independent features and target variable.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified splitting to preserve class proportions.
- Handled missing values using median imputation.
- Used Scikit-learn Pipelines to keep preprocessing consistent and prevent data leakage during cross-validation.

## 5. Features Used

The final model uses eight numerical features:

| Feature | Description |
|---|---|
| `koi_period` | Orbital period of the candidate |
| `koi_depth` | Transit depth |
| `koi_duration` | Transit duration |
| `koi_prad` | Estimated planetary radius |
| `koi_impact` | Impact parameter |
| `koi_steff` | Effective temperature of the host star |
| `koi_srad` | Radius of the host star |
| `koi_model_snr` | Transit signal-to-noise ratio |

## 6. Models Implemented

Several machine learning experiments were conducted:

1. **Logistic Regression:** Used as the baseline classification model.
2. **Random Forest Classifier:** Introduced to capture nonlinear relationships between features.
3. **Class-weighted Random Forest:** Used to account for class imbalance.
4. **Hyperparameter-tuned Random Forest:** Optimized using RandomizedSearchCV with Macro F1-score as the selection metric.

## 7. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion matrix
- Stratified 5-fold cross-validation

Macro F1-score was used during model selection to give equal importance to all three classification categories.

### Final Model Performance

The selected hyperparameter-tuned Random Forest achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 77.21% |
| Macro F1-score | Approximately 0.74 |

### Class-wise F1-scores

| Class | F1-score |
|---|---:|
| CANDIDATE | 0.56 |
| CONFIRMED | 0.80 |
| FALSE POSITIVE | 0.85 |

### Cross-validation Results

Before hyperparameter tuning, the initial eight-feature Random Forest achieved:

- Mean accuracy: 75.49%
- Mean Macro F1-score: 70.52%

The relatively small variation across folds suggested consistent performance.

## 8. Feature Importance

Feature importance was examined to understand which inputs contributed most to the Random Forest's predictions.

The transit signal-to-noise ratio (`koi_model_snr`) was the most important feature in the initial eight-feature Random Forest experiment.

Feature importance indicates how the model uses features for its predictions; it does not establish causation.

## 9. Streamlit Application

The trained model was integrated into a Streamlit application.

The application allows users to:

- Enter planetary and host-star measurements.
- Submit the measurements to the trained model.
- Receive a predicted classification.
- Learn about the project through an About section.

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 11. Future Improvements

Potential improvements for future versions include:

- Experimenting with additional relevant features.
- Improving classification performance for candidate exoplanets.
- Exploring additional machine learning algorithms.
- Working with raw Kepler light-curve data.
- Adding transit visualization and more interactive analysis.

## 12. Project Status

**Version 1:** Model development and local Streamlit application completed. Cloud deployment is in progress.

---

This project was developed as a hands-on machine learning project to understand the complete workflow of a supervised classification problem, from exploratory data analysis and preprocessing to model evaluation and application development.
