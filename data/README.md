# Data

This directory contains the synthetic energy-system dataset used in the project.

## Dataset File

- `synthetic_energy_data.csv`

The dataset contains 5,000 observations:

- 4,000 normal observations
- 1,000 anomalous observations

## Input Features

- `voltage_v`
- `current_a`
- `frequency_hz`
- `temperature_c`
- `dc_link_voltage_v`
- `thd_percent`
- `active_power_w`

## Labels

- `label = 0`: normal operation
- `label = 1`: anomalous operation

The `anomaly_type` column describes the simulated anomaly scenario.

## Important Note

The dataset is synthetic and is intended only for proof-of-concept experimentation. It should not be interpreted as a validated representation of a real energy system.