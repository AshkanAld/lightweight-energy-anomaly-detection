from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "voltage_v",
    "current_a",
    "frequency_hz",
    "temperature_c",
    "dc_link_voltage_v",
    "thd_percent",
    "active_power_w",
]

TARGET_COLUMN = "label"


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_dataset() -> pd.DataFrame:
    """Load the synthetic energy-system dataset."""

    dataset_path = (
        get_project_root()
        / "data"
        / "synthetic_energy_data.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            "Run: python src/generate_data.py"
        )

    dataset = pd.read_csv(dataset_path)

    return dataset


def validate_dataset(dataset: pd.DataFrame) -> None:
    """Check that all required columns exist."""

    required_columns = FEATURE_COLUMNS + [
        "anomaly_type",
        TARGET_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean duplicate, missing, and invalid observations."""

    cleaned_dataset = dataset.copy()

    # Remove exact duplicate rows.
    cleaned_dataset = cleaned_dataset.drop_duplicates()

    # Replace infinite values with missing values.
    cleaned_dataset = cleaned_dataset.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # Fill missing numerical values using column medians.
    for column in FEATURE_COLUMNS:
        cleaned_dataset[column] = pd.to_numeric(
            cleaned_dataset[column],
            errors="coerce",
        )

        median_value = cleaned_dataset[column].median()

        cleaned_dataset[column] = (
            cleaned_dataset[column]
            .fillna(median_value)
        )

    # Ensure that the target contains only 0 and 1.
    cleaned_dataset[TARGET_COLUMN] = pd.to_numeric(
        cleaned_dataset[TARGET_COLUMN],
        errors="coerce",
    )

    cleaned_dataset = cleaned_dataset.dropna(
        subset=[TARGET_COLUMN]
    )

    cleaned_dataset[TARGET_COLUMN] = (
        cleaned_dataset[TARGET_COLUMN]
        .astype(int)
    )

    valid_labels = {0, 1}

    observed_labels = set(
        cleaned_dataset[TARGET_COLUMN].unique()
    )

    if not observed_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected target labels: {observed_labels}"
        )

    return cleaned_dataset.reset_index(drop=True)


def split_dataset(
    dataset: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split features and labels into training and test sets."""

    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    return x_train, x_test, y_train, y_test


def print_dataset_summary(
    original_dataset: pd.DataFrame,
    cleaned_dataset: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Print a concise preprocessing summary."""

    print("=" * 60)
    print("DATASET PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Original rows: {len(original_dataset)}")
    print(f"Cleaned rows:  {len(cleaned_dataset)}")
    print(
        "Removed rows:  "
        f"{len(original_dataset) - len(cleaned_dataset)}"
    )

    print("\nMissing values after cleaning:")
    print(
        cleaned_dataset[
            FEATURE_COLUMNS + [TARGET_COLUMN]
        ]
        .isna()
        .sum()
    )

    print("\nComplete class distribution:")
    print(
        cleaned_dataset[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    print("\nTraining set:")
    print(f"Features shape: {x_train.shape}")
    print(
        y_train.value_counts()
        .sort_index()
    )

    print("\nTest set:")
    print(f"Features shape: {x_test.shape}")
    print(
        y_test.value_counts()
        .sort_index()
    )


def main() -> None:
    original_dataset = load_dataset()

    validate_dataset(original_dataset)

    cleaned_dataset = clean_dataset(
        original_dataset
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = split_dataset(cleaned_dataset)

    print_dataset_summary(
        original_dataset=original_dataset,
        cleaned_dataset=cleaned_dataset,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()