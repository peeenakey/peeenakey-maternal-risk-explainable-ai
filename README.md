# 🩺 Maternal Health Risk Predictor & Explainable AI (XAI)

## 📌 Overview
An end-to-end machine learning pipeline that predicts maternal health risks based on patient vitals. Instead of acting as a "black box," this project uses Explainable AI (SHAP) to show exactly which medical factors (like blood sugar or blood pressure) drive each risk prediction.

## 🎯 Real-World Impact
In healthcare, doctors cannot blindly trust an AI's diagnosis. This project bridges the gap between high accuracy and human trust by providing interpretable, patient-level explanations for clinical predictions.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

## 📊 Key Results
* **Predictive Accuracy:** Achieved ~80% accuracy in classifying patients into Low, Mid, and High-risk categories.
* **Feature Importance:** Identified Blood Sugar as the leading indicator of elevated maternal risk.
* **Model Transparency:** Generated SHAP summary plots to decode the model's decision-making process for individual patient profiles.

## 📸 Visual Insights
*(Note: Upload your images to your repo and add the links here!)*
* `![Confusion Matrix](link_to_confusion_matrix.png)` - *Shows model accuracy across risk levels.*
* `![SHAP Analysis](link_to_shap_summary_bar.png)` - *Illustrates how each feature impacts the final risk score.*

## 🚀 How to Run This Project
1. Clone the repository.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn shap matplotlib seaborn`
3. Run the Jupyter Notebook or Python script to view the data pipeline, train the model, and generate the XAI visualizations.
