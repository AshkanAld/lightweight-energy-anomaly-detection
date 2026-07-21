from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import (
    FEATURE_COLUMNS,
    clean_dataset,
    load_dataset,
    validate_dataset,
)


def get_results_directory() -> Path:
    """Create and return the results directory."""

    project_root = Path(__file__).resolve().parent.parent
    results_directory = project_root / "results"

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return results_directory


def plot_class_distribution(
    dataset: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot normal and anomalous class counts."""

    class_counts = (
        dataset["label"]
        .value_counts()
        .sort_index()
    )

    class_names = ["Normal", "Anomaly"]

    plt.figure(figsize=(7, 5))

    plt.bar(
        class_names,
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0),
        ],
    )

    plt.title("Class Distribution")
    plt.xlabel("Operating condition")
    plt.ylabel("Number of observations")

    for index, value in enumerate(
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0),
        ]
    ):
        plt.text(
            index,
            value,
            str(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = (
        output_directory
        / "class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_anomaly_type_distribution(
    dataset: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot the number of samples for each anomaly type."""

    anomaly_counts = (
        dataset["anomaly_type"]
        .value_counts()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        anomaly_counts.index,
        anomaly_counts.values,
    )

    plt.title("Distribution of Operating Conditions")
    plt.xlabel("Number of observations")
    plt.ylabel("Condition type")

    plt.tight_layout()

    output_path = (
        output_directory
        / "anomaly_type_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_feature_distributions(
    dataset: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot each feature separately for normal and anomalous data."""

    normal_data = dataset[
        dataset["label"] == 0
    ]

    anomaly_data = dataset[
        dataset["label"] == 1
    ]

    number_of_features = len(FEATURE_COLUMNS)

    figure, axes = plt.subplots(
        number_of_features,
        1,
        figsize=(10, 4 * number_of_features),
    )

    for axis, feature in zip(
        axes,
        FEATURE_COLUMNS,
    ):
        axis.hist(
            normal_data[feature],
            bins=40,
            alpha=0.6,
            label="Normal",
            density=True,
        )

        axis.hist(
            anomaly_data[feature],
            bins=40,
            alpha=0.6,
            label="Anomaly",
            density=True,
        )

        axis.set_title(
            f"Distribution of {feature}"
        )

        axis.set_xlabel(feature)
        axis.set_ylabel("Density")
        axis.legend()

    figure.tight_layout()

    output_path = (
        output_directory
        / "feature_distributions.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_correlation_matrix(
    dataset: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot the correlation matrix of numerical features."""

    correlation_columns = (
        FEATURE_COLUMNS + ["label"]
    )

    correlation_matrix = (
        dataset[correlation_columns]
        .corr()
    )

    figure, axis = plt.subplots(
        figsize=(11, 9)
    )

    image = axis.imshow(
        correlation_matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    axis.set_xticks(
        np.arange(
            len(correlation_columns)
        )
    )

    axis.set_yticks(
        np.arange(
            len(correlation_columns)
        )
    )

    axis.set_xticklabels(
        correlation_columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        correlation_columns
    )

    for row_index in range(
        len(correlation_columns)
    ):
        for column_index in range(
            len(correlation_columns)
        ):
            correlation_value = (
                correlation_matrix.iloc[
                    row_index,
                    column_index,
                ]
            )

            axis.text(
                column_index,
                row_index,
                f"{correlation_value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Pearson correlation",
    )

    axis.set_title(
        "Feature Correlation Matrix"
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "correlation_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    dataset = load_dataset()

    validate_dataset(dataset)

    dataset = clean_dataset(dataset)

    output_directory = (
        get_results_directory()
    )

    plot_class_distribution(
        dataset,
        output_directory,
    )

    plot_anomaly_type_distribution(
        dataset,
        output_directory,
    )

    plot_feature_distributions(
        dataset,
        output_directory,
    )

    plot_correlation_matrix(
        dataset,
        output_directory,
    )

    print("\nAll visualizations were generated successfully.")


if __name__ == "__main__":
    main()