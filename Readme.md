# Lightweight Energy Anomaly Detection

A reproducible Python project for detecting simulated faults and cyber-attacks in cyber-physical energy systems using lightweight machine-learning methods.

## Project Motivation

Modern cyber-physical energy systems, including power converters, electric-vehicle chargers, smart-grid components, and industrial electrical infrastructure, rely on sensor measurements and embedded control platforms.

Faulty sensors, abnormal operating conditions, and manipulated measurements may lead to unreliable control decisions or physical damage. This project investigates lightweight machine-learning methods for identifying such conditions.

## Current Project Scope

The first version of the project uses a synthetically generated dataset representing normal and anomalous operating conditions.

The simulated measurements include:

- AC voltage
- Current
- Grid frequency
- Equipment temperature
- DC-link voltage
- Total harmonic distortion
- Active power

The simulated anomaly scenarios include:

- Voltage sag
- Overvoltage
- Current spike
- Temperature anomaly
- Sensor-bias attack
- False-data injection

## Project Structure

```text
lightweight-energy-anomaly-detection/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic_energy_data.csv
├── src/
│   └── generate_data.py
└── results/

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/lightweight-energy-anomaly-detection.git
cd lightweight-energy-anomaly-detection
