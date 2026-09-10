# Task 6: K-Nearest Neighbors (KNN) Classification

**Internship:** AI & ML Internship (Elevate Labs)
**Objective:** Understand and implement KNN for classification problems.
**Tools:** Scikit-learn, Pandas, Matplotlib
**Dataset:** Iris Dataset (`data/Iris.csv`) — 150 samples, 3 classes (Setosa, Versicolor, Virginica), 4 numeric features.

## What was done

1. **Load & normalize** — Loaded `Iris.csv`, dropped the `Id` column, and standardized the four numeric
   features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) using `StandardScaler`
   so that all features contribute equally to distance calculations.
2. **Train/test split** — 80/20 stratified split to keep class balance in both sets.
3. **Model** — Used `KNeighborsClassifier` from `sklearn.neighbors`.
4. **K experimentation** — Trained and evaluated the model for K = 1 to 20, plotted accuracy vs. K, and
   picked the best-performing K.
5. **Evaluation** — Computed accuracy, confusion matrix, and a full classification report
   (precision/recall/F1) on the held-out test set for the best K.
6. **Decision boundary visualization** — Trained a 2D KNN model on `PetalLengthCm` and `PetalWidthCm`
   (the two most discriminative features) and plotted the resulting decision regions with the training
   points overlaid.

## Files

```
knn_task/
├── knn_iris.py              # Main script: run with `python knn_iris.py`
├── data/
│   └── Iris.csv              # Dataset
├── outputs/
│   ├── accuracy_vs_k.png     # Accuracy for K = 1..20
│   ├── confusion_matrix.png  # Confusion matrix heatmap for the best K
│   ├── decision_boundary.png # 2D decision boundary (Petal Length vs Petal Width)
│   └── evaluation_report.txt # Accuracy, confusion matrix, classification report (text)
└── README.md
```

## How to run

```bash
pip install scikit-learn pandas matplotlib numpy
python knn_iris.py
```

## Results

- **Best K found:** 1 (accuracy plateaus at ~0.967 for several odd K values too, e.g. K=7, 9–13, 15–20)
- **Test accuracy:** 96.7% (29/30 correct on the held-out test set)
- **Confusion matrix:** All 10 Setosa and all 10 Versicolor test samples were classified correctly;
  1 of 10 Virginica samples was misclassified as Versicolor — expected, since these two species
  overlap slightly in petal measurements.

See `outputs/evaluation_report.txt` for the full numeric report.

## Interview Questions & Answers

**1. How does the KNN algorithm work?**
KNN is an instance-based (lazy) learning algorithm. It stores the entire training dataset and, to
classify a new point, computes the distance (usually Euclidean) from that point to every training
point, finds the K closest ones ("neighbors"), and assigns the class that is most common among those
K neighbors (majority vote). There's no explicit training/model-fitting phase — all the work happens
at prediction time.

**2. How do you choose the right K?**
Common approaches: (a) try a range of K values and pick the one with the best validation/cross-validation
accuracy (as done here, K=1..20), (b) use odd K for binary classification to avoid ties, (c) as a rule
of thumb start near K = √n (n = number of training samples) and tune from there. Small K → low bias but
high variance (sensitive to noise/overfitting); large K → smoother boundaries but risk of underfitting
and including irrelevant/far-away points.

**3. Why is normalization important in KNN?**
KNN relies directly on distance calculations. If features are on different scales (e.g., one feature
ranges 0–1 and another 0–10000), the large-scale feature will dominate the distance metric and
effectively drown out the others, regardless of their actual predictive importance. Normalizing
(standardizing or min-max scaling) puts all features on a comparable scale so each contributes fairly.

**4. What is the time complexity of KNN?**
Training is essentially O(1) (just storing the data). Prediction for one query point with brute-force
search is O(n·d), where n is the number of training samples and d is the number of features, since you
compute the distance to every training point. For m query points that becomes O(m·n·d). This is why
KNN can be slow at inference time on large datasets; spatial index structures like KD-Trees or Ball
Trees (used internally by sklearn) can reduce this to roughly O(d·log n) per query in lower dimensions.

**5. What are the pros and cons of KNN?**
*Pros:* simple and intuitive; no training phase; naturally handles multi-class problems; non-parametric
(makes no assumption about the underlying data distribution); can model complex/non-linear decision
boundaries.
*Cons:* slow at prediction time for large datasets; sensitive to irrelevant features and feature scale;
sensitive to noisy data and outliers; suffers from the "curse of dimensionality" in high-dimensional
spaces; requires storing the entire dataset in memory.

**6. Is KNN sensitive to noise?**
Yes, especially with small K (e.g., K=1), since a single mislabeled or noisy neighbor can directly
determine the prediction. Increasing K averages over more neighbors, which generally makes the model
more robust to noise, at the cost of possibly blurring the true decision boundary.

**7. How does KNN handle multi-class problems?**
KNN handles multi-class classification naturally, with no modification needed. It simply looks at the
class labels of the K nearest neighbors (whatever the number of classes) and returns the majority
class among them — the same voting mechanism works for 2 classes or 20 classes.

**8. What's the role of distance metrics in KNN?**
The distance metric defines what "nearest" means, so it directly determines which points are considered
neighbors and therefore the model's predictions. Euclidean distance is the default and most common
choice for continuous numeric features, but other metrics exist: Manhattan distance (sum of absolute
differences, often used with sparse/high-dimensional data), Minkowski distance (a generalization of
Euclidean/Manhattan), Hamming distance (for categorical/binary features), and cosine similarity (for
text/high-dimensional vectors where direction matters more than magnitude). Choosing an appropriate
metric for the data type and scale is critical to KNN's performance.
