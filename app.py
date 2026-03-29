import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── STYLES ───────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1B3A6B; }
    .metric-label { font-size: 0.82rem; color: #555; margin-top: 4px; }
    .risk-high { background: #ffe0e0; border-radius: 12px; padding: 20px; text-align: center; }
    .risk-low  { background: #e0ffe0; border-radius: 12px; padding: 20px; text-align: center; }
    .section-title { font-size: 1.15rem; font-weight: 600; color: #1B3A6B; margin: 18px 0 8px 0; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & PREPARE DATA ──────────────────────────────────────
@st.cache_data
def load_and_train():
    # Pima Indians Diabetes Dataset — built-in via URL
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, names=cols)

    # Data cleaning — replace 0s with median for medical columns
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in medical_cols:
        df[col] = df[col].replace(0, df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train 3 models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, preds) * 100, 1),
            "Precision": round(precision_score(y_test, preds) * 100, 1),
            "Recall":    round(recall_score(y_test, preds) * 100, 1),
            "F1 Score":  round(f1_score(y_test, preds) * 100, 1),
        }
        trained[name] = model

    best_name = max(results, key=lambda k: results[k]["F1 Score"])
    best_model = trained[best_name]

    return df, X.columns.tolist(), scaler, trained, results, best_name, best_model, X_test_sc, y_test

df, feature_cols, scaler, trained_models, results, best_name, best_model, X_test_sc, y_test = load_and_train()

# ── HEADER ───────────────────────────────────────────────────
st.title("🏥 Disease Risk Predictor")
st.markdown("Predicts **diabetes risk** based on health metrics using Machine Learning.")
st.markdown("Dataset: Pima Indians Diabetes Dataset (Kaggle) — 768 patients, 8 features")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Risk", "📊 Model Performance", "📈 Data Insights"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Patient Health Metrics")
    st.markdown("Adjust the sliders and click **Predict** to get the risk assessment.")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose     = st.slider("Glucose Level (mg/dL)", 50, 200, 120)
        blood_press = st.slider("Blood Pressure (mmHg)", 30, 130, 70)
        skin_thick  = st.slider("Skin Thickness (mm)", 5, 60, 20)

    with col2:
        insulin     = st.slider("Insulin Level (IU/mL)", 10, 500, 80)
        bmi         = st.slider("BMI", 15.0, 55.0, 25.0, step=0.1)
        dpf         = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.5, step=0.01)
        age         = st.slider("Age", 18, 90, 33)

    st.markdown("---")

    if st.button("🔍 Predict Diabetes Risk", width='stretch'):
        input_data = np.array([[pregnancies, glucose, blood_press, skin_thick,
                                insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction   = best_model.predict(input_scaled)[0]
        probability  = best_model.predict_proba(input_scaled)[0][1] * 100

        st.markdown("---")
        st.markdown("### Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h2>⚠️ HIGH RISK</h2>
                    <p>Diabetes risk detected</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h2>✅ LOW RISK</h2>
                    <p>No diabetes risk detected</p>
                </div>""", unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{probability:.1f}%</div>
                <div class="metric-label">Risk Probability</div>
            </div>""", unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_name.split()[0]}</div>
                <div class="metric-label">Model Used ({best_name})</div>
            </div>""", unsafe_allow_html=True)

        # Feature importance
        st.markdown("#### Which factors influenced this prediction most?")
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        else:
            importances = np.abs(best_model.coef_[0])
            importances = importances / importances.sum()

        fi_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#1B3A6B" if v == fi_df["Importance"].max() else "#A8BFDD"
                  for v in fi_df["Importance"]]
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("⚠️ This is a machine learning prediction tool for educational purposes only. Always consult a qualified medical professional for health decisions.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Comparison")
    st.markdown(f"**Best Model Selected: {best_name}** (based on highest F1 Score)")
    st.markdown("---")

    # Metrics table
    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ["Model", "Accuracy %", "Precision %", "Recall %", "F1 Score %"]
    st.dataframe(results_df, width='stretch', hide_index=True)

    st.markdown("---")

    # Bar chart comparison
    st.markdown("#### Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    model_names = list(results.keys())
    accuracies  = [results[m]["Accuracy"] for m in model_names]
    colors = ["#1B3A6B" if m == best_name else "#A8BFDD" for m in model_names]
    bars = ax.bar(model_names, accuracies, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Confusion matrix for best model
    st.markdown(f"#### Confusion Matrix — {best_name}")
    y_pred = best_model.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Total Patients", "768"),
        (c2, "Features", "8"),
        (c3, "Diabetic", f"{df['Outcome'].sum()}"),
        (c4, "Non-Diabetic", f"{(df['Outcome']==0).sum()}")
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### Glucose Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[df["Outcome"]==0]["Glucose"].hist(ax=ax, alpha=0.6, color="#A8BFDD", label="No Diabetes", bins=20)
        df[df["Outcome"]==1]["Glucose"].hist(ax=ax, alpha=0.6, color="#1B3A6B", label="Diabetes", bins=20)
        ax.set_xlabel("Glucose Level")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col6:
        st.markdown("#### Age Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[df["Outcome"]==0]["Age"].hist(ax=ax, alpha=0.6, color="#A8BFDD", label="No Diabetes", bins=20)
        df[df["Outcome"]==1]["Age"].hist(ax=ax, alpha=0.6, color="#1B3A6B", label="Diabetes", bins=20)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("🔍 View Raw Dataset (first 50 rows)"):
        st.dataframe(df.head(50), width='stretch')

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with Python · Pandas · Scikit-learn · Matplotlib · Streamlit"
    " &nbsp;|&nbsp; Project by Shahid Siraj S</small></center>",
    unsafe_allow_html=True
)
