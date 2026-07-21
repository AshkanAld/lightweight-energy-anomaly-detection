# Source Code

This directory contains the complete experimental pipeline.

## Files

- `generate_data.py`: generates the synthetic energy-system dataset
- `preprocess.py`: validates, cleans, and splits the dataset
- `visualize.py`: creates exploratory data visualizations
- `train_logistic_regression.py`: trains and evaluates Logistic Regression
- `train_random_forest.py`: trains and evaluates Random Forest
- `train_isolation_forest.py`: trains and evaluates Isolation Forest
- `compare_models.py`: compares predictive performance and efficiency
- `analyze_models.py`: analyzes confusion matrices, feature importance, inference time, and model size

## Execution

Run all scripts from the root directory of the repository. For example:

```bash
python src/train_random_forest.py