from pathlib import Path

import json
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import (
    clean_dataset,
    load_dataset,
    split_dataset,
    validate_dataset,
)


RANDOM_SEED = 42
CONTAMINATION = 0.20


def get_directories() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent.parent

    models_directory = project_root / "models"
    results_directory = project_root / "results"

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return models_directory, results_directory


def convert_predictions(
    isolation_predictions,
):
    """
    Convert Isolation Forest output:
    1  -> 0 (normal)
    -1 -> 1 (anomaly)
    """

    return [
        0 if prediction == 1 else 1
        for prediction in isolation_predictions
    ]


def calculate_metrics(
    y_true,
    y_predicted,
    anomaly_scores,
) -> dict:
    confusion = confusion_matrix(
        y_true,
        y_predicted,
    )

    true_negative = int(confusion[0, 0])
    false_positive = int(confusion[0, 1])
    false_negative = int(confusion[1, 0])
    true_positive = int(confusion[1, 1])

    false_negative_rate = (
        false_negative
        / (false_negative + true_positive)
        if false_negative + true_positive > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
        if false_positive + true_negative > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_predicted,
            )
        ),
        "precision_anomaly": float(
            precision_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "recall_anomaly": float(
            recall_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "f1_anomaly": float(
            f1_score(
                y_true,
                y_predicted,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                anomaly_scores,
            )
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def measure_inference_time(
    model,
    x_test: pd.DataFrame,
    number_of_repeats: int = 100,
) -> float:
    start_time = time.perf_counter()

    for _ in range(number_of_repeats):
        model.predict(x_test)

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    total_predictions = (
        len(x_test)
        * number_of_repeats
    )

    return (
        elapsed_time
        / total_predictions
    ) * 1000


def save_confusion_matrix(
    y_true,
    y_predicted,
    results_directory: Path,
) -> None:
    confusion = confusion_matrix(
        y_true,
        y_predicted,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=[
            "Normal",
            "Anomaly",
        ],
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        "Isolation Forest Confusion Matrix"
    )

    figure.tight_layout()

    output_path = (
        results_directory
        / "isolation_forest_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_roc_curve(
    y_true,
    anomaly_scores,
    roc_auc: float,
    results_directory: Path,
) -> None:
    false_positive_rates, true_positive_rates, _ = (
        roc_curve(
            y_true,
            anomaly_scores,
        )
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        false_positive_rates,
        true_positive_rates,
        label=f"Isolation Forest (AUC = {roc_auc:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Isolation Forest ROC Curve")
    plt.legend()
    plt.tight_layout()

    output_path = (
        results_directory
        / "isolation_forest_roc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    dataset = load_dataset()
    validate_dataset(dataset)
    dataset = clean_dataset(dataset)

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = split_dataset(dataset)

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                IsolationForest(
                    n_estimators=200,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training Isolation Forest...")

    training_start = time.perf_counter()

    # Labels are not used here.
    model.fit(x_train)

    training_time_seconds = (
        time.perf_counter()
        - training_start
    )

    raw_predictions = model.predict(
        x_test
    )

    predicted_labels = convert_predictions(
        raw_predictions
    )

    # Smaller decision_function values indicate
    # stronger anomaly behaviour, so the sign is reversed.
    anomaly_scores = -model.decision_function(
        x_test
    )

    metrics = calculate_metrics(
        y_true=y_test,
        y_predicted=predicted_labels,
        anomaly_scores=anomaly_scores,
    )

    metrics["training_time_seconds"] = float(
        training_time_seconds
    )

    metrics[
        "average_inference_time_ms_per_sample"
    ] = float(
        measure_inference_time(
            model,
            x_test,
        )
    )

    (
        models_directory,
        results_directory,
    ) = get_directories()

    model_path = (
        models_directory
        / "isolation_forest.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    model_size_bytes = model_path.stat().st_size

    metrics["model_size_bytes"] = int(
        model_size_bytes
    )

    metrics["model_size_kilobytes"] = float(
        model_size_bytes / 1024
    )

    metrics["contamination"] = float(
        CONTAMINATION
    )

    metrics_path = (
        results_directory
        / "isolation_forest_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    save_confusion_matrix(
        y_true=y_test,
        y_predicted=predicted_labels,
        results_directory=results_directory,
    )

    save_roc_curve(
        y_true=y_test,
        anomaly_scores=anomaly_scores,
        roc_auc=metrics["roc_auc"],
        results_directory=results_directory,
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predicted_labels,
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    print("\nIsolation Forest metrics:")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(
                f"{metric_name}: "
                f"{metric_value:.6f}"
            )
        else:
            print(
                f"{metric_name}: "
                f"{metric_value}"
            )

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()