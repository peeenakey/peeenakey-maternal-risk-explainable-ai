# Maternal Risk Prediction using Explainable AI

## Overview

This project focuses on predicting maternal health risk levels using clinical parameters and machine learning, with an emphasis on interpretability through Explainable AI (XAI). The system classifies patients into Low Risk, Mid Risk, and High Risk categories. A Random Forest model is used for prediction, and SHAP (SHapley Additive exPlanations) is applied to explain feature contributions, making the model suitable for healthcare applications where transparency is critical.

## Problem Statement

Maternal health complications remain a significant challenge, especially in resource-constrained environments. Early detection of high-risk pregnancies can reduce maternal mortality, enable timely intervention, and assist clinicians in decision-making. This project builds a data-driven pipeline to analyze maternal health indicators, predict risk level, and explain model decisions.

## Dataset

The dataset includes the following clinical features:

* Age
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Blood Sugar (BS)
* Body Temperature
* Heart Rate
* Risk Level (Target Variable)

### Target Encoding

* Low Risk → 0
* Mid Risk → 1
* High Risk → 2

## Project Pipeline

### 1. Data Preprocessing

* Removed duplicate entries
* Handled anomalies (e.g., invalid heart rate values)
* Missing values imputed using median
* Data validation and consistency checks

### 2. Exploratory Data Analysis (EDA)

* Feature distributions using histograms
* Risk-level comparison using boxplots
* Correlation heatmap to identify relationships

Generated visualizations:

* feature_distribution.png
* features_vs_risk.png
* correlation_heatmap.png

### 3. Model Training

* Model: Random Forest Classifier
* Train-test split: 80-20
* Random state: 42

### 4. Model Evaluation

* Accuracy Score
* Confusion Matrix

Output:

* confusion_matrix.png

### 5. Explainable AI (XAI)

* Feature importance using Random Forest
* SHAP for global interpretability

Outputs:

* feature_importances.png
* shap_summary_bar.png

## Results

The model successfully classifies maternal risk levels with strong accuracy. Key influencing features include blood pressure (systolic and diastolic), blood sugar, and heart rate. SHAP analysis confirms how each feature contributes to predictions, improving interpretability and trust.

## Tech Stack

Programming Language:

* Python

Libraries Used:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* shap

## Installation

git clone https://github.com/your-username/maternal-risk-explainable-ai.git
cd maternal-risk-explainable-ai
pip install -r requirements.txt

Additional dependency:
pip install shap

## Usage

python maternal-risk-explainable-ai.py

Or run in Google Colab for interactive execution.

## Project Structure

maternal-risk-explainable-ai/
│
├── data/
│   └── archive.zip
│
├── outputs/
│   ├── feature_distribution.png
│   ├── features_vs_risk.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── feature_importances.png
│   └── shap_summary_bar.png
│
├── maternal-risk-explainable-ai.py
├── README.md
└── requirements.txt

## Key Insights

* Blood pressure and blood sugar are strong indicators of maternal risk
* Data preprocessing significantly improves model performance
* Explainability is essential for healthcare AI adoption

## Limitations

* Dataset size and diversity may limit generalization
* No real-time deployment
* Requires clinical validation

## Future Work

* Deploy as a web or mobile application
* Integrate wearable sensor data
* Use advanced models (XGBoost, deep learning)
* Add personalized patient-level explanations
* Validate with real-world clinical datasets

## Applications

* Clinical decision support systems
* Rural healthcare monitoring
* Telemedicine platforms
* Maternal health tracking applications

## License

This project is intended for academic and research purposes.

## Author

Developed as part of a healthcare AI project focusing on interpretable machine learning for maternal risk assessment.
