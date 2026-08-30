# Engineering Design & Architecture Log

## 1. Data Ingestion & Signal Conditioning

- **Missing Telemetry Handling**: Sensor telemetry demonstrates temporal continuity with discrete sampling dropouts. A dual-stage strategy is implemented: linear time-series interpolation for short dropouts ($\le 6$ hours) to preserve local gradients, followed by causal forward propagation for extended outages. Mean or median imputation across the entire series was avoided to prevent dampening hydrological seasonality.
- **Anomaly Mitigation**: Rather than dropping records and introducing temporal gaps into the autoregressive structure, IQR winsorization ($3.0 \times \text{IQR}$) is applied to clip non-physical voltage spikes, sensor biofouling, and transmission errors while maintaining an unbroken time index.

## 2. Feature Engineering Rationale

- **Autoregressive Lags**: 1h, 3h, 6h, 12h, and 24h lags capture short-term catchment momentum and diurnal flushing dynamics.
- **Rolling Moments**: Moving averages, standard deviations, and min/max envelopes over 3h, 6h, 12h, and 24h horizons capture variance shifts and localized trends.
- **Cyclical Time Decomposition**: Hour-of-day, day-of-week, and month-of-year are mapped to orthogonal trigonometric components ($\sin/\cos$) to maintain cyclical continuity without artificial boundary jumps between 23:00 and 00:00.
- **Physical Domain Proxies**: Nitrate-to-conductivity ratios and turbidity-temperature interaction indices model runoff transport physics and biological algal bloom dynamics.

## 3. Modeling Methodology & Validation

- **Algorithm**: XGBoost Regressors were selected over recurrent architectures (LSTM/GRU) due to lower inference latency on edge deployments, sample efficiency on tabular telemetry, and interpretability via split-gain importance distributions.
- **Validation Protocol**: 5-fold expanding walk-forward cross-validation ensures strict chronological evaluation without future lookahead bias.

## 4. Advisory Engine & Reliability

- **Context Grounding**: The advisory layer injects real-time telemetry, 6-hour forecast vectors, and guideline standards into a structured prompt contract that strictly forbids ungrounded assertions.
- **Deterministic Redundancy**: In offline environments or upon API connection timeouts, a rule-based engine executes threshold compliance checks and issues standard operational directives automatically.
