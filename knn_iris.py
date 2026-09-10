"""
Task 6: K-Nearest Neighbors (KNN) Classification
Dataset: Iris Dataset (Iris.csv)

Steps:
1. Load dataset and normalize features
2. Train KNeighborsClassifier from sklearn
3. Experiment with different values of K
4. Evaluate using accuracy and confusion matrix
5. Visualize decision boundaries
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1. Load dataset and normalize features
# -----------------------------------------------------------------
df = pd.read_csv("data/Iris.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Drop the Id column (not a feature)
df = df.drop(columns=["Id"])

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].values
y = df["Species"].values

# Encode class labels to integers for convenience
classes = sorted(df["Species"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y_encoded = np.array([class_to_idx[label] for label in y])

# Train/test split (stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalize features (KNN is distance-based, so scaling matters a lot)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# 2 & 3. Train KNeighborsClassifier and experiment with different K
# -----------------------------------------------------------------
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

best_k = list(k_values)[int(np.argmax(accuracies))]
best_acc = max(accuracies)
print(f"\nBest K: {best_k} with test accuracy: {best_acc:.4f}")

# Plot accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), accuracies, marker="o", color="#4C72B0")
plt.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"Best K = {best_k}")
plt.title("KNN Accuracy vs. K (on all 4 features, normalized)")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Test Accuracy")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/accuracy_vs_k.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# 4. Final model evaluation using accuracy and confusion matrix
# -----------------------------------------------------------------
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred = final_knn.predict(X_test_scaled)

final_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=classes)

print(f"\nFinal Model (K={best_k}) Test Accuracy: {final_acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Save confusion matrix plot
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title(f"Confusion Matrix (K={best_k})")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# Save text report to file
with open(f"{OUT_DIR}/evaluation_report.txt", "w") as f:
    f.write(f"Best K: {best_k}\n")
    f.write(f"Final Test Accuracy: {final_acc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)

# -----------------------------------------------------------------
# 5. Visualize decision boundaries
# Using two features (Petal Length & Petal Width) since they separate
# the Iris species best and allow a 2D plot.
# -----------------------------------------------------------------
feat_x, feat_y = "PetalLengthCm", "PetalWidthCm"
X2 = df[[feat_x, feat_y]].values
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(X2_train_scaled, y2_train)

# Build mesh grid
h = 0.02
x_min, x_max = X2_train_scaled[:, 0].min() - 1, X2_train_scaled[:, 0].max() + 1
y_min, y_max = X2_train_scaled[:, 1].min() - 1, X2_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = matplotlib.colors.ListedColormap(["#FFDDDD", "#DDFFDD", "#DDDDFF"])
cmap_bold = ["#FF0000", "#00AA00", "#0000FF"]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

for idx, cls in enumerate(classes):
    mask = y2_train == idx
    plt.scatter(
        X2_train_scaled[mask, 0], X2_train_scaled[mask, 1],
        c=cmap_bold[idx], label=cls, edgecolor="k", s=40,
    )

plt.title(f"KNN Decision Boundary (K={best_k})\nFeatures: {feat_x} & {feat_y} (standardized)")
plt.xlabel(f"{feat_x} (standardized)")
plt.ylabel(f"{feat_y} (standardized)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/decision_boundary.png", dpi=150)
plt.close()

print("\nAll outputs saved to the 'outputs/' folder.")
