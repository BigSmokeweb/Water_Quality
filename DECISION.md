# Decision Log (DECISION.md)

### 1. Data Cleaning, Features & Model Architecture
- **Cleaning**: Short sensor dropouts ($\le 6\text{h}$) are filled via linear interpolation to maintain temporal gradients; longer outages use forward propagation. Non-physical telemetry spikes are clipped using $3.0 \times \text{IQR}$ winsorization rather than row deletion, preserving an unbroken time series.
- **Features**: Lagged observations (1h–24h) and rolling statistics (mean, std, min, max over 3h–24h windows) capture catchment momentum and variance. Cyclical temporal dynamics (hour, day, month) are encoded via $\sin/\cos$ pairs, alongside domain interaction proxies (e.g., Turbidity $\times$ Water Temp).
- **Model & Validation**: Multi-step forecasting is powered by **XGBoost Regressors**, chosen for high tabular sample efficiency, rapid inference, and robust non-linear gradient boosting. Models are evaluated using a 5-fold expanding walk-forward cross-validation to strictly prevent lookahead data leakage.

### 2. LLM Orchestration & Hallucination Prevention
- **Grounded Prompt Contract**: The Gemini LLM pipeline is strictly bounded by deterministic context injection (exact sensor telemetry, model forecast vectors, and safety guideline thresholds). The prompt enforces strict instruction boundaries that forbid ungrounded speculation.
- **Deterministic Fallback**: If the API is unreachable, a hardcoded rule engine verifies compliance against safety limits and issues standardized operational directives.

