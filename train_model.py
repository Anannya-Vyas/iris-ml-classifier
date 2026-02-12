import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- LOAD DATA ---------------- #
iris = load_iris()
X = iris.data
y = iris.target

# ---------------- TRAIN TEST SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN MODEL ---------------- #
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------- METRICS ---------------- #
test_accuracy = model.score(X_test, y_test)

cv_scores = cross_val_score(model, X, y, cv=5)
cv_mean = np.mean(cv_scores)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    output_dict=True
)

# ---------------- SAVE EVERYTHING ---------------- #
with open("iris_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
    
        "cv_accuracy": cv_mean,
        "confusion_matrix": conf_matrix,
        "feature_importance": model.feature_importances_,
        "class_names": iris.target_names,
        "classification_report": report
    }, f)

print("Model saved successfully.")

print("Cross Validation Accuracy:", round(cv_mean * 100, 2), "%")
