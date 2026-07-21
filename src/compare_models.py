from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


MODEL_FILES = {
    "Logistic Regression": "logistic_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "Isolation Forest": "isolation_forest_metrics.json",
}


def load_metrics(results_directory: Path) -> pd.DataFrame:
    """
    Load model metrics from JSON files and return
    them as a pandas DataFrame.
    """

    rows = []

    for model_name, filename in MODEL_FILES.items():
        metrics_path = results_directory / filename

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics file: {metrics_path}"
            )

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as metrics_file:
            metrics = json.load(metrics_file)

        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision_anomaly": metrics[
                    "precision_anomaly"
                ],
                "recall_anomaly": metrics[
                    "recall_anomaly"
                ],
                "f1_anomaly": metrics[
                    "f1_anomaly"
                ],
                "roc_auc": metrics["roc_auc"],
                "false_negative_rate": metrics[
                    "false_negative_rate"
                ],
                "false_positive_rate": metrics[
                    "false_positive_rate"
                ],
                "inference_time_ms": metrics[
                    "average_inference_time_ms_per_sample"
                ],
                "model_size_kb": metrics[
                    "model_size_kilobytes"
                ],
            }
        )

    return pd.DataFrame(rows)


def save_performance_comparison_chart(
    comparison: pd.DataFrame,
    results_directory: Path,
) -> None:
    """
    Save a bar chart comparing the main predictive
    performance metrics of all models.
    """

    selected_metrics = (
        comparison
        .set_index("model")
        [
            [
                "accuracy",
                "precision_anomaly",
                "recall_anomaly",
                "f1_anomaly",
                "roc_auc",
            ]
        ]
    )

    figure, axis = plt.subplots(
        figsize=(13, 8)
    )

    selected_metrics.plot(
        kind="bar",
        ax=axis,
        width=0.8,
    )

    axis.set_title(
        "Model Performance Comparison",
        fontsize=16,
        pad=15,
    )

    axis.set_xlabel(
        "Model",
        fontsize=12,
    )

    axis.set_ylabel(
        "Metric value",
        fontsize=12,
    )

    axis.set_ylim(
        0,
        1.05,
    )

    axis.tick_params(
        axis="x",
        rotation=0,
        labelsize=11,
    )

    axis.tick_params(
        axis="y",
        labelsize=10,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    axis.legend(
        title="Metric",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        frameon=True,
    )

    figure.tight_layout()

    output_path = (
        results_directory
        / "model_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved chart: {output_path}")


def save_efficiency_comparison_chart(
    comparison: pd.DataFrame,
    results_directory: Path,
) -> None:
    """
    Save a second chart comparing model size and
    inference time.
    """

    efficiency_data = comparison.set_index(
        "model"
    )[
        [
            "inference_time_ms",
            "model_size_kb",
        ]
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
    )

    efficiency_data[
        "inference_time_ms"
    ].plot(
        kind="bar",
        ax=axes[0],
    )

    axes[0].set_title(
        "Average Inference Time per Sample"
    )

    axes[0].set_xlabel("")
    axes[0].set_ylabel("Milliseconds")
    axes[0].tick_params(
        axis="x",
        rotation=0,
    )

    axes[0].grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    efficiency_data[
        "model_size_kb"
    ].plot(
        kind="bar",
        ax=axes[1],
    )

    axes[1].set_title(
        "Model Size Comparison"
    )

    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Kilobytes")
    axes[1].tick_params(
        axis="x",
        rotation=0,
    )

    axes[1].grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    figure.tight_layout()

    output_path = (
        results_directory
        / "model_efficiency_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved chart: {output_path}")


def print_comparison_summary(
    comparison: pd.DataFrame,
) -> None:
    """
    Print the complete comparison table and the
    best model for selected criteria.
    """

    print("=" * 110)
    print("MODEL COMPARISON")
    print("=" * 110)

    print(
        comparison.to_string(
            index=False,
        )
    )

    best_recall_row = comparison.loc[
        comparison[
            "recall_anomaly"
        ].idxmax()
    ]

    print("\nBest anomaly recall:")

    print(
        f"{best_recall_row['model']}: "
        f"{best_recall_row['recall_anomaly']:.4f}"
    )

    lowest_fnr_row = comparison.loc[
        comparison[
            "false_negative_rate"
        ].idxmin()
    ]

    print("\nLowest false-negative rate:")

    print(
        f"{lowest_fnr_row['model']}: "
        f"{lowest_fnr_row['false_negative_rate']:.4f}"
    )

    smallest_model_row = comparison.loc[
        comparison[
            "model_size_kb"
        ].idxmin()
    ]

    print("\nSmallest model:")

    print(
        f"{smallest_model_row['model']}: "
        f"{smallest_model_row['model_size_kb']:.2f} KB"
    )

    fastest_model_row = comparison.loc[
        comparison[
            "inference_time_ms"
        ].idxmin()
    ]

    print("\nFastest inference:")

    print(
        f"{fastest_model_row['model']}: "
        f"{fastest_model_row['inference_time_ms']:.6f} ms/sample"
    )

    best_unsupervised_row = comparison[
        comparison["model"] == "Isolation Forest"
    ].iloc[0]

    print("\nUnsupervised model summary:")

    print(
        "Isolation Forest achieved "
        f"{best_unsupervised_row['recall_anomaly']:.4f} "
        "anomaly recall without using labels during training."
    )


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    results_directory = (
        project_root
        / "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = load_metrics(
        results_directory
    )

    comparison_path = (
        results_directory
        / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    save_performance_comparison_chart(
        comparison=comparison,
        results_directory=results_directory,
    )

    save_efficiency_comparison_chart(
        comparison=comparison,
        results_directory=results_directory,
    )

    print_comparison_summary(
        comparison
    )

    print(
        f"\nComparison table saved to: "
        f"{comparison_path}"
    )


if __name__ == "__main__":
    main()