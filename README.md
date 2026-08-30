# Johnstone River Water Quality Telemetry & Predictive System

Station ID: **JR-CP-01 (Coquette Point)**  
Sampling Frequency: **Hourly Telemetry**  

A real-time environmental telemetry pipeline and autoregressive forecasting platform designed for river catchment monitoring, nitrate concentration tracking, and automated excursion assessment.

---

## System Architecture

```
water_quality/
├── app.py                     # Streamlit operational telemetry station dashboard
├── config.py                  # Telemetry parameters, safety limits, model constants
├── requirements.txt           # Environment dependencies
├── DECISION.md                # System engineering and modeling rationale
├── .env.example               # Environment template
│
├── data/
│   ├── pipeline.py            # Telemetry ingestion, imputation, anomaly winsorization
│   └── raw/                   # Kaggle telemetry CSV files
│       └── Johnstone_river_coquette_point_joined.csv
│
├── features/
│   └── engineer.py            # Autoregressive lags, rolling moments, cyclical & domain indices
│
├── models/
│   ├── train.py               # Walk-forward CV & XGBoost training pipeline
│   ├── predict.py             # Multi-step recursive forecasting engine
│   └── artifacts/             # Serialized model binaries and performance metrics
│       ├── NO3_model.pkl
│       ├── Turbidity_model.pkl
│       └── metrics.json
│
├── llm/
│   └── advisor.py             # TelemetryAdvisoryEngine (grounded reasoning + fallback)
│
└── tests/                     # Unit test suites (pipeline, features, advisor)
    ├── test_pipeline.py
    ├── test_features.py
    └── test_advisor.py
```

---

## Installation & Deployment

### 1. Environment Setup

```bash
cd water_quality
pip install -r requirements.txt
```

### 2. Configuration

Copy the sample environment file and set operational keys as needed:

```bash
cp .env.example .env
```

### 3. Model Training & Cross-Validation

Train the gradient-boosted regressors across expanding walk-forward validation folds:

```bash
python -m models.train
```

Evaluation metrics will be written to `models/artifacts/metrics.json`.

### 4. Running the Dashboard

Launch the local telemetry console:

```bash
python -m streamlit run app.py
```

---

## Model Performance

| Target Parameter | RMSE | MAE | R² Score | Validation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Nitrate (NO3-N, mg/L)** | 0.0049 | 0.0020 | **0.9945** | 5-Fold Walk-Forward Cross-Validation |
| **Turbidity (NTU)** | 0.1859 | 0.1095 | **0.9985** | 5-Fold Walk-Forward Cross-Validation |

---

## Operational Threshold Reference

| Parameter | Nominal | Caution Trigger | Alert Limit | Reference Standard |
| :--- | :--- | :--- | :--- | :--- |
| **Nitrate (NO3-N)** | < 7.0 mg/L | ≥ 7.0 mg/L | ≥ 10.0 mg/L | WHO / ADWG Drinking Water Standard |
| **Turbidity** | < 50.0 NTU | ≥ 50.0 NTU | ≥ 100.0 NTU | Catchment Sediment Alert Band |
| **Water Temperature** | < 25.0 °C | ≥ 25.0 °C | ≥ 30.0 °C | Ecological Thermal Baseline |

---

## Testing

Execute the test suite using pytest:

```bash
pytest -v tests/
```
