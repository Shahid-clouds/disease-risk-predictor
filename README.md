# 🏥 Disease Risk Predictor

A machine learning web app that predicts diabetes risk based on patient health metrics.
Built with Python, Pandas, Scikit-learn, and Streamlit.

## 🚀 Features

- **Risk Prediction** — Input health metrics and get instant diabetes risk prediction
- **Probability Score** — Shows confidence percentage of the prediction
- **Feature Importance** — Shows which health factors influenced the prediction most
- **Model Comparison** — Compares Logistic Regression, Decision Tree, and Random Forest
- **Confusion Matrix** — Visual model performance breakdown
- **Data Insights** — EDA charts including glucose and age distribution by outcome
- **Correlation Heatmap** — Feature relationships visualized

## 🛠️ Tech Stack

- Python
- Pandas (data manipulation)
- Scikit-learn (ML models, preprocessing, evaluation)
- Matplotlib & Seaborn (visualizations)
- Streamlit (web app + deployment)
- NumPy (numerical operations)

## 📊 Dataset

Pima Indians Diabetes Dataset (Kaggle)
- 768 patient records
- 8 features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- Target: Diabetic (1) or Non-Diabetic (0)

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Best model selected automatically based on highest F1 Score.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Live Demo

[View on Streamlit Cloud](#) ← add your link after deploying

## ⚠️ Disclaimer

This app is for educational purposes only. Always consult a qualified medical professional for health decisions.

## 👤 Author

Shahid Siraj S
[GitHub](https://github.com/Shahid-clouds) | [LinkedIn](https://linkedin.com/in/shahidsiraj)
