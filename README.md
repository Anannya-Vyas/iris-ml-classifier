# 🌸 Iris ML Classifier (Deployed)

A machine learning web application that classifies Iris flower species using a Random Forest model with proper validation and evaluation metrics.

🔗 **Live Demo:** [https://iris-ml-classifier-anannyavyas.streamlit.app/](https://iris-ml-classifier-anannyavyas.streamlit.app/)

---

## 🚀 What This Project Demonstrates

- **End-to-end ML pipeline** from data loading to deployment
- **Proper train-test split** for unbiased evaluation
- **5-Fold Cross Validation** for robust performance estimation
- **Confusion Matrix analysis** for detailed error analysis
- **Precision, Recall, F1-score evaluation** beyond simple accuracy
- **Feature importance interpretation** for model explainability
- **PCA visualization** for dimensionality reduction and data exploration
- **Model persistence** using Pickle for reproducibility
- **Cloud deployment** using Streamlit for real-world accessibility

---

## 📊 Model Performance

- **Cross-Validation Accuracy:** ~96–100%
- **High class-wise precision & recall** across all three Iris species
- **Petal features** identified as dominant predictors
- **Zero overfitting** demonstrated through consistent train-test performance

---

## 🧠 Key Technical Decisions

### Why Random Forest?
Robust ensemble model that reduces overfitting through bagging, handles feature interactions effectively, and provides built-in feature importance metrics for interpretability.

### Why Cross-Validation?
Ensures the model generalizes well across different data splits. Performance is not dependent on a single train-test split, providing more reliable accuracy estimates.

### Why Feature Importance?
Interprets model behavior and identifies which features drive predictions. This transparency is crucial for understanding model decisions and building trust in production systems.

### Why PCA Visualization?
Visualizes class separability in lower-dimensional space (2D/3D), helping to understand the dataset structure and validate that different species are indeed distinguishable.

---

## 🎯 Why This Matters

While the Iris dataset is simple, this project demonstrates:
- Proper model validation techniques used in real-world ML systems
- Evaluation metrics beyond accuracy (precision, recall, F1-score)
- Model interpretability and explainability
- Deployment practices required in production environments
- Best practices for reproducible machine learning

These are the fundamentals that scale to complex production ML systems.

---

## 🛠 Tech Stack

**Languages & Libraries:**
- Python 3.8+
- Scikit-learn (ML modeling)
- NumPy & Pandas (data manipulation)
- Plotly (interactive visualizations)
- Streamlit (web deployment)

**Tools:**
- Git (version control)
- Pickle (model serialization)
- GitHub (code hosting)

---

## 📂 Project Structure

```
iris-ml-classifier/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training and evaluation script
├── iris_model.pkl          # Trained Random Forest model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore             # Git ignore file
```

---

## 🔄 Reproducibility

### Clone the Repository
```bash
git clone https://github.com/Anannya-Vyas/iris-ml-classifier.git
cd iris-ml-classifier
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train the Model
```bash
python train_model.py
```

This will:
- Load and preprocess the Iris dataset
- Train a Random Forest classifier
- Perform 5-fold cross-validation
- Display evaluation metrics (accuracy, precision, recall, F1-score)
- Show confusion matrix
- Display feature importance
- Save the trained model as `iris_model.pkl`

### Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🌐 Deployment

This application is deployed on **Streamlit Community Cloud**, providing free hosting for Streamlit apps connected to GitHub repositories.

**Deployment Steps:**
1. Push code to GitHub repository
2. Connect repository to Streamlit Community Cloud
3. Configure `requirements.txt` for dependencies
4. Deploy and share the public URL

---

## 📈 Features

### Interactive Prediction
- Input sepal and petal measurements through sliders
- Real-time prediction with probability scores
- Visual feedback for predicted species

### Model Evaluation
- Confusion matrix heatmap
- Classification metrics (precision, recall, F1-score)
- Cross-validation scores
- Feature importance bar chart

### Data Visualization
- 2D PCA projection of the dataset
- 3D PCA visualization for spatial understanding
- Interactive plots using Plotly

---

## 🎓 Learning Outcomes

Through this project, I gained hands-on experience with:
- Building end-to-end ML pipelines
- Implementing proper model evaluation techniques
- Understanding ensemble methods (Random Forest)
- Creating interactive web applications with Streamlit
- Deploying ML models to production
- Writing clean, reproducible code
- Professional documentation practices

---

## 🔮 Future Enhancements

- [ ] Add support for other classification algorithms (SVM, XGBoost, Neural Networks)
- [ ] Implement A/B testing for model comparison
- [ ] Add data augmentation techniques
- [ ] Create API endpoints using FastAPI
- [ ] Add user authentication and prediction history
- [ ] Implement model monitoring and retraining pipeline
- [ ] Add unit tests and CI/CD pipeline

---


## 👤 Author

**Anannya Vyas**

- GitHub: [@Anannya-Vyas](https://github.com/Anannya-Vyas)


---

##  Acknowledgments

- Iris dataset from UCI Machine Learning Repository
- Streamlit community for deployment platform
- Scikit-learn documentation and tutorials

---

**⭐ If you found this project helpful, please consider giving it a star!**
