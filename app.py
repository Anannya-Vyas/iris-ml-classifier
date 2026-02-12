import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Iris ML Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------- DARK THEME ---------------- #
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
.stButton>button {
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
    background-color: #6C63FF;
    color: white;
    border: none;
}
.stButton>button:hover {
    background-color: #5848e5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL DATA ---------------- #
with open("iris_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
accuracy = data["cv_accuracy"]
conf_matrix = data["confusion_matrix"]
feature_importance = data["feature_importance"]
class_names = data["class_names"]
classification_report_data = data["classification_report"]

# ---------------- HEADER ---------------- #
st.title("🌸 Iris Flower Classifier")
st.caption("Machine Learning Web Application | Random Forest Model")
st.divider()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📊 Model Details")
st.sidebar.metric("Cross-Validation Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset: Iris (150 samples)")

st.sidebar.divider()
st.sidebar.header("👩‍💻 About Me")
st.sidebar.write("Python & ML Developer")
st.sidebar.write("GitHub: https://github.com/Anannya-Vyas")
st.sidebar.write("PyPI: https://pypi.org/project/stattools-anannya/")

# ---------------- MODEL EVALUATION ---------------- #
st.subheader("📊 Model Evaluation")

# -------- Confusion Matrix -------- #
st.write("### 🔍 Confusion Matrix")

fig_cm = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=class_names,
    y=class_names,
    colorscale="Blues",
    showscale=True
))

fig_cm.update_layout(
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label",
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white")
)

st.plotly_chart(fig_cm, use_container_width=True)

# -------- Feature Importance -------- #
st.write("### 📈 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

fig_imp = go.Figure(data=[
    go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation='h'
    )
])

fig_imp.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white")
)

st.plotly_chart(fig_imp, use_container_width=True)

# -------- PCA Visualization -------- #
st.write("### 🧠 PCA Visualization (Decision Space)")

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig_pca = go.Figure()

for i, class_name in enumerate(class_names):
    fig_pca.add_trace(go.Scatter(
        x=X_pca[y == i, 0],
        y=X_pca[y == i, 1],
        mode='markers',
        name=class_name
    ))

fig_pca.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white"),
    xaxis_title="Principal Component 1",
    yaxis_title="Principal Component 2"
)

# -------- Classification Report -------- #
st.write("### 📄 Classification Report")

report_df = pd.DataFrame(classification_report_data).transpose()

st.dataframe(
    report_df.style.format("{:.2f}"),
    use_container_width=True
)

st.divider()

# ---------------- INPUT SECTION ---------------- #
st.subheader("🌼 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2, step=0.1)

st.divider()

# ---------------- PREDICTION ---------------- #
if st.button("Predict"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_species = class_names[prediction]
    confidence = np.max(probabilities) * 100

    # -------- Prediction Card -------- #
    st.subheader("🔎 Prediction Result")

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    ">
        <h1 style="color:white; margin-bottom:10px;">
            🌸 {predicted_species.upper()}
        </h1>
        <p style="color:white; font-size:20px;">
            Confidence: {confidence:.2f}%
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------- Probability Chart -------- #
    fig_prob = go.Figure()

    fig_prob.add_trace(go.Bar(
        x=class_names,
        y=probabilities,
        text=[f"{p*100:.1f}%" for p in probabilities],
        textposition='auto'
    ))

    fig_prob.update_layout(
        title="Class Probability Distribution",
        yaxis_title="Probability",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white")
    )
 
    st.plotly_chart(fig_prob, use_container_width=True)

    # -------- Plot User Input on PCA -------- #
    input_pca = pca.transform(input_data)

    fig_pca.add_trace(go.Scatter(
        x=[input_pca[0][0]],
        y=[input_pca[0][1]],
        mode='markers',
        marker=dict(size=16, symbol="x"),
        name="Your Input"
    ))

    st.plotly_chart(fig_pca, use_container_width=True)

st.divider()
st.caption("Built with Python, Scikit-learn, Streamlit & Plotly")
