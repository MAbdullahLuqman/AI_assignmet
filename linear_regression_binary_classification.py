# ============================================
# AI ASSIGNMENT 3
# Linear Regression for Binary Classification
# Student Performance Dataset
# ============================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# --------------------------------------------
# 2. Load Dataset
# --------------------------------------------
df = pd.read_csv("../dataset/student_mat.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# --------------------------------------------
# 3. Create Binary Target Variable
# --------------------------------------------
median_grade = df["G3"].median()

df["above_median_performance"] = np.where(
    df["G3"] > median_grade, 1, 0
)

# Drop original grade column
df.drop("G3", axis=1, inplace=True)

# --------------------------------------------
# 4. Encode Categorical Variables
# --------------------------------------------
df = pd.get_dummies(df, drop_first=True)

# --------------------------------------------
# 5. Split Features and Target
# --------------------------------------------
X = df.drop("above_median_performance", axis=1)
y = df["above_median_performance"]

# --------------------------------------------
# 6. Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# 7. Train Linear Regression Model
# --------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------
# 8. Predictions
# --------------------------------------------
y_pred_continuous = model.predict(X_test)

# Convert to binary using threshold
y_pred_binary = np.where(y_pred_continuous >= 0.5, 1, 0)

# --------------------------------------------
# 9. Regression Evaluation
# --------------------------------------------
mse = mean_squared_error(y_test, y_pred_continuous)
r2 = r2_score(y_test, y_pred_continuous)

print("\nRegression Metrics")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# --------------------------------------------
# 10. Classification Evaluation
# --------------------------------------------
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("\nClassification Metrics")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# --------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------
cm = confusion_matrix(y_test, y_pred_binary)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------
# 12. ROC Curve and AUC
# --------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_continuous)
auc = roc_auc_score(y_test, y_pred_continuous)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --------------------------------------------
# 13. Metrics Comparison Bar Chart
# --------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("Classification Metrics Comparison")
plt.show()
