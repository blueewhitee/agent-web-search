"""Stage 3.5 scrubber internal data types (D-010).

These are NOT part of the API response — internal to the scrubber service,
same pattern as `app/schemas/fetch.py` (internal `FetchResult`).
"""

from dataclasses import dataclass, field
from enum import Enum


class ThreatType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    IMPERSONATION = "impersonation"
    XSS_INJECTION = "xss_injection"
    PAYLOAD_SMUGGLING = "payload_smuggling"


@dataclass
class ThreatDetection:
    threat_type: ThreatType
    confidence: float
    pattern_id: str  # short label, e.g. "inject:ignore_instructions" — NOT raw regex


@dataclass
class ScrubResult:
    clean: bool
    content: str
    threats: list[ThreatDetection] = field(default_factory=list)
    risk_score: float = 0.0
    redactions: int = 0
