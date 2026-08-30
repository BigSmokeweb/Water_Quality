import json
import logging
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GEMINI_API_KEY, GEMINI_MODEL, SAFETY_THRESHOLDS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an automated Water Quality Advisory System analyzing river telemetry.
Ground all responses strictly on the provided real-time telemetry and 6-hour forecast values.
Do not hallucinate readings, speculate on unmeasured contaminants, or cite values not present in the context.

Operational guideline thresholds:
- Nitrate (NO3-N): Caution >= 7.0 mg/L | Unsafe >= 10.0 mg/L
- Turbidity: Caution >= 50.0 NTU | Unsafe >= 100.0 NTU
- Water Temperature: Caution >= 25.0 °C | Unsafe >= 30.0 °C

Response Contract:
1. Provide an executive summary of current parameter statuses (Normal / Caution / Alert).
2. Detail projected threshold excursions within the 6-hour horizon.
3. State unambiguous operational recommendations for municipal, agricultural, or environmental stakeholders.
4. Maintain a formal, concise technical tone.
5. If queries deviate from water quality analysis, indicate the query is outside operational scope."""


def format_context_payload(current_telemetry: Dict[str, Any], forecast_payload: Dict[str, Any]) -> str:
    """Format telemetry and forecast data into JSON context for evaluation."""
    context = {
        "telemetry_readings": current_telemetry,
        "forecast_series": {
            k: v for k, v in forecast_payload.items() if k != "timestamps"
        },
        "forecast_timestamps": forecast_payload.get("timestamps", []),
        "operational_thresholds": {
            k: {
                "caution_limit": v["caution"],
                "alert_limit": v["safe"],
                "unit": v["unit"],
            }
            for k, v in SAFETY_THRESHOLDS.items()
        },
    }
    return json.dumps(context, indent=2)


def generate_rule_based_advisory(current_telemetry: Dict[str, Any], forecast_payload: Dict[str, Any]) -> str:
    """Deterministic rule-based safety evaluation used in offline or fallback modes."""
    sections = ["### Automated Telemetry Assessment\n"]
    worst_status = "NORMAL"

    for param, spec in SAFETY_THRESHOLDS.items():
        val = current_telemetry.get(param)
        if val is None:
            continue

        if val >= spec["safe"]:
            status_label = "[ALERT - EXCEEDED LIMIT]"
            worst_status = "ALERT"
        elif val >= spec["caution"]:
            status_label = "[CAUTION - ELEVATED]"
            if worst_status != "ALERT":
                worst_status = "CAUTION"
        else:
            status_label = "[NORMAL]"

        sections.append(f"- **{spec['name']}**: {val:.2f} {spec['unit']} {status_label}")

    forecast_alerts = []
    for param, spec in SAFETY_THRESHOLDS.items():
        predictions = forecast_payload.get(param, [])
        for hour_idx, pred_val in enumerate(predictions):
            if pred_val >= spec["safe"]:
                forecast_alerts.append(
                    f"{spec['name']} projected at {pred_val:.2f} {spec['unit']} at T+{hour_idx + 1}h"
                )
                break

    if forecast_alerts:
        sections.append(f"\n**Horizon Excursion Warnings:**\n- " + "\n- ".join(forecast_alerts))

    sections.append("\n**Operational Directives:**")
    if worst_status == "ALERT":
        sections.append("Suspend non-essential intake and notify regional catchment monitoring authorities.")
    elif worst_status == "CAUTION":
        sections.append("Increase telemetry sampling frequency and verify upstream discharge compliance.")
    else:
        sections.append("Parameters remain within nominal operating bands. Maintain baseline monitoring.")

    return "\n".join(sections)


class TelemetryAdvisoryEngine:
    """Advisory engine supporting both LLM-grounded analysis and deterministic fallback logic."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL):
        self._api_key = api_key or GEMINI_API_KEY
        self._model_name = model_name
        self._client = None
        self._chat_session = None
        self._llm_active = False

        if self._api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                self._chat_session = self._client.chats.create(
                    model=self._model_name,
                    config={"system_instruction": SYSTEM_PROMPT},
                )
                self._llm_active = True
                logger.info("Advisory engine initialized with model '%s'", self._model_name)
            except Exception as exc:
                logger.warning("LLM client initialization failed: %s. Using deterministic fallback.", exc)
        else:
            logger.info("No API key configured. Operating in deterministic rule-based mode.")

    @property
    def is_llm_available(self) -> bool:
        return self._llm_active

    def evaluate(
        self,
        query: str,
        current_telemetry: Dict[str, Any],
        forecast_data: Dict[str, Any],
    ) -> str:
        """Generate evaluation grounded in sensor telemetry and forecasts."""
        context_str = format_context_payload(current_telemetry, forecast_data)
        prompt_payload = (
            f"TELEMETRY_DATA_CONTEXT:\n{context_str}\n\n"
            f"OPERATOR_QUERY: {query}"
        )

        if self._llm_active and self._chat_session:
            try:
                response = self._chat_session.send_message(prompt_payload)
                if response.text:
                    return response.text
            except Exception as exc:
                logger.error("LLM inference error: %s. Falling back to rule-based engine.", exc)

        return generate_rule_based_advisory(current_telemetry, forecast_data)

    def reset(self) -> None:
        """Reset conversation state."""
        if self._llm_active and self._client:
            try:
                self._chat_session = self._client.chats.create(
                    model=self._model_name,
                    config={"system_instruction": SYSTEM_PROMPT},
                )
            except Exception as exc:
                logger.error("Failed to reset session: %s", exc)


# Backwards-compatible alias
WaterQualityAdvisor = TelemetryAdvisoryEngine
