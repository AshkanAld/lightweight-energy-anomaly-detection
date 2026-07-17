# Results

This directory contains the current model-evaluation outputs and exploratory data visualizations.

## Exploratory Data Analysis

- `class_distribution.png`: normal versus anomalous observations
- `anomaly_type_distribution.png`: distribution of simulated operating conditions
- `feature_distributions.png`: comparison of feature distributions for normal and anomalous samples
- `correlation_matrix.png`: correlations among numerical features and the binary label

## Logistic Regression Baseline

- `logistic_confusion_matrix.png`: confusion matrix on the test set
- `logistic_roc_curve.png`: ROC curve and area under the curve
- `logistic_metrics.json`: classification metrics, inference time, and model size

The Logistic Regression model is used as a lightweight and interpretable baseline for binary anomaly detection.
