from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_SAMPLES = 5000
ANOMALY_RATIO = 0.20


def generate_normal_data(number_of_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate simulated normal operating measurements."""

    voltage = rng.normal(loc=230.0, scale=3.0, size=number_of_samples)
    current = rng.normal(loc=10.0, scale=1.2, size=number_of_samples)
    frequency = rng.normal(loc=50.0, scale=0.04, size=number_of_samples)
    temperature = rng.normal(loc=42.0, scale=3.0, size=number_of_samples)
    dc_link_voltage = rng.normal(loc=700.0, scale=8.0, size=number_of_samples)
    thd = np.clip(
        rng.normal(loc=2.5, scale=0.6, size=number_of_samples),
        a_min=0.1,
        a_max=None,
    )

    active_power = voltage * current * rng.normal(
        loc=0.92,
        scale=0.02,
        size=number_of_samples,
    )

    return pd.DataFrame(
        {
            "voltage_v": voltage,
            "current_a": current,
            "frequency_hz": frequency,
            "temperature_c": temperature,
            "dc_link_voltage_v": dc_link_voltage,
            "thd_percent": thd,
            "active_power_w": active_power,
            "anomaly_type": "normal",
            "label": 0,
        }
    )


def apply_voltage_sag(data: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator) -> None:
    data.loc[indices, "voltage_v"] *= rng.uniform(0.55, 0.80, size=len(indices))
    data.loc[indices, "current_a"] *= rng.uniform(1.15, 1.50, size=len(indices))
    data.loc[indices, "anomaly_type"] = "voltage_sag"


def apply_overvoltage(data: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator) -> None:
    data.loc[indices, "voltage_v"] *= rng.uniform(1.12, 1.30, size=len(indices))
    data.loc[indices, "anomaly_type"] = "overvoltage"


def apply_current_spike(data: pd.DataFrame, indices: np.ndarray, rng: np.random.Generator) -> None:
    data.loc[indices, "current_a"] *= rng.uniform(1.8, 3.0, size=len(indices))
    data.loc[indices, "temperature_c"] += rng.uniform(5.0, 15.0, size=len(indices))
    data.loc[indices, "anomaly_type"] = "current_spike"


def apply_temperature_anomaly(
    data: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> None:
    data.loc[indices, "temperature_c"] += rng.uniform(18.0, 40.0, size=len(indices))
    data.loc[indices, "anomaly_type"] = "temperature_anomaly"


def apply_sensor_bias_attack(
    data: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> None:
    data.loc[indices, "voltage_v"] += rng.uniform(15.0, 35.0, size=len(indices))
    data.loc[indices, "dc_link_voltage_v"] += rng.uniform(25.0, 60.0, size=len(indices))
    data.loc[indices, "anomaly_type"] = "sensor_bias_attack"


def apply_false_data_injection(
    data: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> None:
    data.loc[indices, "frequency_hz"] += rng.choice(
        [-1.0, 1.0],
        size=len(indices),
    ) * rng.uniform(0.3, 1.2, size=len(indices))

    data.loc[indices, "thd_percent"] += rng.uniform(4.0, 12.0, size=len(indices))
    data.loc[indices, "anomaly_type"] = "false_data_injection"


def recalculate_active_power(data: pd.DataFrame, rng: np.random.Generator) -> None:
    power_factor = rng.normal(loc=0.92, scale=0.025, size=len(data))
    power_factor = np.clip(power_factor, 0.75, 1.0)

    data["active_power_w"] = (
        data["voltage_v"] * data["current_a"] * power_factor
    )


def generate_dataset() -> pd.DataFrame:
    """Generate the complete normal and anomalous energy-system dataset."""

    rng = np.random.default_rng(RANDOM_SEED)

    dataset = generate_normal_data(NUMBER_OF_SAMPLES, rng)

    anomaly_count = int(NUMBER_OF_SAMPLES * ANOMALY_RATIO)
    anomaly_indices = rng.choice(
        dataset.index,
        size=anomaly_count,
        replace=False,
    )

    anomaly_groups = np.array_split(anomaly_indices, 6)

    apply_voltage_sag(dataset, anomaly_groups[0], rng)
    apply_overvoltage(dataset, anomaly_groups[1], rng)
    apply_current_spike(dataset, anomaly_groups[2], rng)
    apply_temperature_anomaly(dataset, anomaly_groups[3], rng)
    apply_sensor_bias_attack(dataset, anomaly_groups[4], rng)
    apply_false_data_injection(dataset, anomaly_groups[5], rng)

    dataset.loc[anomaly_indices, "label"] = 1

    recalculate_active_power(dataset, rng)

    dataset = dataset.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return dataset


def save_dataset(dataset: pd.DataFrame) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "data"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "synthetic_energy_data.csv"
    dataset.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    dataset = generate_dataset()
    output_path = save_dataset(dataset)

    print(f"Dataset saved to: {output_path}")
    print(f"Number of rows: {len(dataset)}")
    print("\nClass distribution:")
    print(dataset["label"].value_counts())
    print("\nAnomaly distribution:")
    print(dataset["anomaly_type"].value_counts())
    print("\nFirst five rows:")
    print(dataset.head())


if __name__ == "__main__":
    main()