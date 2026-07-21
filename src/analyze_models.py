from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from preprocess import (
    FEATURE_COLUMNS,
    clean_dataset,
    load_dataset,
    split_dataset,
    validate_dataset,
)


MODEL_CONFIG = {
    "Logistic Regression": {
        "model_file": "logistic_regression.pkl",
        "metrics_file": "logistic_metrics.json",
    },
    "Random Forest": {
        "model_file": "random_forest.pkl",
        "metrics_file": "random_forest_metrics.json",
    },
    "Isolation Forest": {
        "model_file": "isolation_forest.pkl",
        "metrics_file": "isolation_forest_metrics.json",
    },
}

NUMBER_OF_REPEATS = 100
WARMUP_REPEATS = 10


def get_project_directories() -> tuple[Path, Path, Path]:
    """Return project, model, and result directories."""

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    models_directory = project_root / "models"
    results_directory = project_root / "results"

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        project_root,
        models_directory,
        results_directory,
    )


def load_saved_models(
    models_directory: Path,
) -> dict:
    """Load all trained models."""

    loaded_models = {}

    for model_name, model_info in MODEL_CONFIG.items():
        model_path = (
            models_directory
            / model_info["model_file"]
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(
            model_path
        )

        print(
            f"Loaded model: "
            f"{model_name}"
        )

    return loaded_models


def convert_isolation_predictions(
    raw_predictions,
) -> np.ndarray:
    """
    Convert Isolation Forest predictions:

    1  means normal and becomes 0.
    -1 means anomaly and becomes 1.
    """

    return np.where(
        raw_predictions == 1,
        0,
        1,
    )


def predict_labels(
    model_name: str,
    model,
    features: pd.DataFrame,
) -> np.ndarray:
    """Generate binary predictions for a model."""

    raw_predictions = model.predict(
        features
    )

    if model_name == "Isolation Forest":
        return convert_isolation_predictions(
            raw_predictions
        )

    return np.asarray(
        raw_predictions
    )


def create_confusion_matrix_summary(
    models: dict,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    results_directory: Path,
) -> pd.DataFrame:
    """Calculate and save confusion-matrix values."""

    rows = []

    for model_name, model in models.items():
        predicted_labels = predict_labels(
            model_name=model_name,
            model=model,
            features=x_test,
        )

        confusion = confusion_matrix(
            y_test,
            predicted_labels,
            labels=[0, 1],
        )

        true_negative = int(
            confusion[0, 0]
        )

        false_positive = int(
            confusion[0, 1]
        )

        false_negative = int(
            confusion[1, 0]
        )

        true_positive = int(
            confusion[1, 1]
        )

        rows.append(
            {
                "model": model_name,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
            }
        )

    summary = pd.DataFrame(rows)

    output_path = (
        results_directory
        / "confusion_matrix_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved confusion-matrix summary: "
        f"{output_path}"
    )

    return summary


def extract_logistic_importance(
    logistic_model,
) -> pd.Series:
    """
    Extract normalized absolute Logistic Regression
    coefficients.
    """

    classifier = logistic_model.named_steps[
        "classifier"
    ]

    absolute_coefficients = np.abs(
        classifier.coef_[0]
    )

    total = absolute_coefficients.sum()

    if total > 0:
        normalized_importance = (
            absolute_coefficients / total
        )
    else:
        normalized_importance = (
            absolute_coefficients
        )

    return pd.Series(
        normalized_importance,
        index=FEATURE_COLUMNS,
        name="Logistic Regression",
    )


def extract_random_forest_importance(
    random_forest_model,
) -> pd.Series:
    """Extract Random Forest feature importance."""

    importance = (
        random_forest_model
        .feature_importances_
    )

    return pd.Series(
        importance,
        index=FEATURE_COLUMNS,
        name="Random Forest",
    )


def create_feature_importance_outputs(
    models: dict,
    results_directory: Path,
) -> pd.DataFrame:
    """
    Compare feature importance for Logistic Regression
    and Random Forest.

    Isolation Forest does not provide an equivalent
    direct feature-importance attribute.
    """

    logistic_importance = (
        extract_logistic_importance(
            models["Logistic Regression"]
        )
    )

    random_forest_importance = (
        extract_random_forest_importance(
            models["Random Forest"]
        )
    )

    importance_table = pd.concat(
        [
            logistic_importance,
            random_forest_importance,
        ],
        axis=1,
    )

    importance_table.index.name = "feature"

    importance_table = (
        importance_table
        .reset_index()
    )

    output_csv = (
        results_directory
        / "feature_importance_comparison.csv"
    )

    importance_table.to_csv(
        output_csv,
        index=False,
    )

    chart_data = (
        importance_table
        .set_index("feature")
    )

    axis = chart_data.plot(
        kind="barh",
        figsize=(11, 7),
    )

    axis.set_title(
        "Feature Importance Comparison"
    )

    axis.set_xlabel(
        "Normalized importance"
    )

    axis.set_ylabel(
        "Input feature"
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        title="Model",
        loc="lower right",
    )

    plt.tight_layout()

    output_chart = (
        results_directory
        / "feature_importance_comparison.png"
    )

    plt.savefig(
        output_chart,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved feature-importance table: "
        f"{output_csv}"
    )

    print(
        f"Saved feature-importance chart: "
        f"{output_chart}"
    )

    return importance_table


def measure_inference_time(
    model_name: str,
    model,
    x_test: pd.DataFrame,
) -> dict:
    """
    Measure batch inference time and average
    inference time per sample.
    """

    # Warm-up runs are excluded from measurement.
    for _ in range(WARMUP_REPEATS):
        predict_labels(
            model_name=model_name,
            model=model,
            features=x_test,
        )

    batch_times_ms = []

    for _ in range(NUMBER_OF_REPEATS):
        start_time = time.perf_counter()

        predict_labels(
            model_name=model_name,
            model=model,
            features=x_test,
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        batch_times_ms.append(
            elapsed_seconds * 1000
        )

    average_batch_time_ms = float(
        np.mean(batch_times_ms)
    )

    median_batch_time_ms = float(
        np.median(batch_times_ms)
    )

    average_time_per_sample_ms = (
        average_batch_time_ms
        / len(x_test)
    )

    median_time_per_sample_ms = (
        median_batch_time_ms
        / len(x_test)
    )

    return {
        "average_batch_time_ms": (
            average_batch_time_ms
        ),
        "median_batch_time_ms": (
            median_batch_time_ms
        ),
        "average_inference_time_ms_per_sample": (
            average_time_per_sample_ms
        ),
        "median_inference_time_ms_per_sample": (
            median_time_per_sample_ms
        ),
    }


def get_model_size(
    model_path: Path,
) -> dict:
    """Calculate saved model size."""

    size_bytes = model_path.stat().st_size

    size_kilobytes = (
        size_bytes / 1024
    )

    size_megabytes = (
        size_kilobytes / 1024
    )

    return {
        "model_size_bytes": int(
            size_bytes
        ),
        "model_size_kb": float(
            size_kilobytes
        ),
        "model_size_mb": float(
            size_megabytes
        ),
    }


def create_deployment_comparison(
    models: dict,
    models_directory: Path,
    x_test: pd.DataFrame,
    results_directory: Path,
) -> pd.DataFrame:
    """
    Compare inference time and saved model size.
    """

    rows = []

    for model_name, model in models.items():
        model_filename = (
            MODEL_CONFIG[model_name]
            ["model_file"]
        )

        model_path = (
            models_directory
            / model_filename
        )

        timing_results = (
            measure_inference_time(
                model_name=model_name,
                model=model,
                x_test=x_test,
            )
        )

        size_results = get_model_size(
            model_path
        )

        row = {
            "model": model_name,
            **timing_results,
            **size_results,
        }

        rows.append(row)

    deployment_comparison = pd.DataFrame(
        rows
    )

    output_csv = (
        results_directory
        / "deployment_comparison.csv"
    )

    deployment_comparison.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"Saved deployment comparison: "
        f"{output_csv}"
    )

    return deployment_comparison


def save_inference_time_chart(
    deployment_comparison: pd.DataFrame,
    results_directory: Path,
) -> None:
    """Save inference-time comparison chart."""

    chart_data = (
        deployment_comparison
        .set_index("model")
        ["median_inference_time_ms_per_sample"]
    )

    axis = chart_data.plot(
        kind="bar",
        figsize=(9, 6),
    )

    axis.set_title(
        "Median Inference Time per Sample"
    )

    axis.set_xlabel(
        "Model"
    )

    axis.set_ylabel(
        "Milliseconds per sample"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    output_path = (
        results_directory
        / "inference_time_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved inference-time chart: "
        f"{output_path}"
    )


def save_model_size_chart(
    deployment_comparison: pd.DataFrame,
    results_directory: Path,
) -> None:
    """Save model-size comparison chart."""

    chart_data = (
        deployment_comparison
        .set_index("model")
        ["model_size_kb"]
    )

    axis = chart_data.plot(
        kind="bar",
        figsize=(9, 6),
    )

    axis.set_title(
        "Saved Model Size Comparison"
    )

    axis.set_xlabel(
        "Model"
    )

    axis.set_ylabel(
        "Model size in kilobytes"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    output_path = (
        results_directory
        / "model_size_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved model-size chart: "
        f"{output_path}"
    )


def print_summary(
    confusion_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    deployment_comparison: pd.DataFrame,
) -> None:
    """Print day-four analysis results."""

    print("\n" + "=" * 100)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 100)

    print(
        confusion_summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE")
    print("=" * 100)

    print(
        feature_importance.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("DEPLOYMENT COMPARISON")
    print("=" * 100)

    print(
        deployment_comparison.to_string(
            index=False
        )
    )

    fastest_model = (
        deployment_comparison.loc[
            deployment_comparison[
                "median_inference_time_ms_per_sample"
            ].idxmin()
        ]
    )

    smallest_model = (
        deployment_comparison.loc[
            deployment_comparison[
                "model_size_kb"
            ].idxmin()
        ]
    )

    print("\nFastest model:")

    print(
        f"{fastest_model['model']}: "
        f"{fastest_model['median_inference_time_ms_per_sample']:.6f} "
        "ms/sample"
    )

    print("\nSmallest model:")

    print(
        f"{smallest_model['model']}: "
        f"{smallest_model['model_size_kb']:.2f} KB"
    )


def main() -> None:
    (
        _,
        models_directory,
        results_directory,
    ) = get_project_directories()

    dataset = load_dataset()

    validate_dataset(
        dataset
    )

    dataset = clean_dataset(
        dataset
    )

    (
        _,
        x_test,
        _,
        y_test,
    ) = split_dataset(
        dataset
    )

    models = load_saved_models(
        models_directory
    )

    confusion_summary = (
        create_confusion_matrix_summary(
            models=models,
            x_test=x_test,
            y_test=y_test,
            results_directory=results_directory,
        )
    )

    feature_importance = (
        create_feature_importance_outputs(
            models=models,
            results_directory=results_directory,
        )
    )

    deployment_comparison = (
        create_deployment_comparison(
            models=models,
            models_directory=models_directory,
            x_test=x_test,
            results_directory=results_directory,
        )
    )

    save_inference_time_chart(
        deployment_comparison=deployment_comparison,
        results_directory=results_directory,
    )

    save_model_size_chart(
        deployment_comparison=deployment_comparison,
        results_directory=results_directory,
    )

    print_summary(
        confusion_summary=confusion_summary,
        feature_importance=feature_importance,
        deployment_comparison=deployment_comparison,
    )

    print(
        "\nDay-four model analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()