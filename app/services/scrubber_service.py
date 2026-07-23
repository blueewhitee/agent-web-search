"""Stage 3.5 prompt-injection scrubber (D-010).

Scans extracted text for injection, exfiltration, impersonation, and XSS
patterns; redacts matches in-place; returns cleaned text. Pure module-level
function (same pattern as `extraction_service.py`) — no class, no state.

Fixes every flaw identified in the agent-search reference (`app/scrubber.py`,
539 LOC): broken risk math, detect/redact mismatch, `re.DOTALL` missing on
XSS, Cyrillic homoglyph false-positive, NFKD-without-NFC corruption, base64
auto-decode mangling, no content-size cap, and false-positive-prone dev-content
patterns.
"""

import re

from app.schemas.scrubber import ScrubResult, ThreatDetection, ThreatType

# -- Stage A: zero-width / RTL char stripping --------------------------------

_INVISIBLE = set('\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff')
_RTL = set('\u202a\u202b\u202c\u202d\u202e')
_STRIP_CHARS = _INVISIBLE | _RTL


def _strip_invisible(text: str) -> tuple[str, int]:
    """Strip zero-width + RTL override chars. Returns (cleaned, count)."""
    stripped = ''.join(c for c in text if c not in _STRIP_CHARS)
    return stripped, len(text) - len(stripped)


# -- Stage B: pattern detection + redaction (same pass) ----------------------

_PATTERNS: list[tuple[re.Pattern, ThreatType, float, str]] = [
    # Injection (15 patterns)
    (re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?(?:your\s+)?instructions"), ThreatType.PROMPT_INJECTION, 0.9, "inject:ignore_instructions"),
    (re.compile(r"(?i)forget\s+(?:all\s+)?(?:your\s+)?(?:previous\s+)?(?:instructions|rules|constraints)"), ThreatType.PROMPT_INJECTION, 0.9, "inject:forget_instructions"),
    (re.compile(r"(?i)system\s*:\s*you\s+are\s+now"), ThreatType.PROMPT_INJECTION, 0.9, "inject:system_override"),
    (re.compile(r"(?i)override\s+(?:your\s+)?(?:previous\s+)?instructions"), ThreatType.PROMPT_INJECTION, 0.9, "inject:override_instructions"),
    (re.compile(r"(?i)disregard\s+(?:all\s+)?(?:previous\s+)?(?:your\s+)?instructions"), ThreatType.PROMPT_INJECTION, 0.9, "inject:disregard_instructions"),
    (re.compile(r"(?i)you\s+are\s+(?:no\s+longer|not)\s+(?:an?\s+)?(?:assistant|ai|agent|bot)"), ThreatType.PROMPT_INJECTION, 0.9, "inject:role_denial"),
    (re.compile(r"(?i)from\s+now\s+on\s+you\s+are"), ThreatType.PROMPT_INJECTION, 0.9, "inject:role_assign"),
    (re.compile(r"(?i)(?:show|tell|give|reveal|print|output|display|share)\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions)"), ThreatType.PROMPT_INJECTION, 0.9, "inject:prompt_leak"),
    (re.compile(r"(?i)(?:dump|export)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|configuration)"), ThreatType.PROMPT_INJECTION, 0.9, "inject:prompt_dump"),
    (re.compile(r"(?i)developer\s+mode"), ThreatType.PROMPT_INJECTION, 0.9, "inject:dev_mode"),
    (re.compile(r"(?i)jailbreak\s+(?:mode|now|the|this|and)"), ThreatType.PROMPT_INJECTION, 0.9, "inject:jailbreak"),
    (re.compile(r"(?i)DAN\s+mode"), ThreatType.PROMPT_INJECTION, 0.9, "inject:dan_mode"),
    (re.compile(r"(?i)bypass\s+(?:all\s+)?(?:restrictions|limitations|safeguards)"), ThreatType.PROMPT_INJECTION, 0.9, "inject:bypass_restrictions"),
    (re.compile(r"(?i)step\s+1\s*:\s*ignore"), ThreatType.PROMPT_INJECTION, 0.85, "inject:chain_step1"),
    (re.compile(r"(?i)first\s*,?\s*ignore"), ThreatType.PROMPT_INJECTION, 0.85, "inject:chain_first"),

    # Exfiltration (5 patterns — dropped environment variables, database
    # credentials, admin/operator key/token, private key, signing key)
    (re.compile(r"(?i)(?:show|tell|give|reveal|share|provide)\s+(?:me\s+)?(?:your|the)\s+(?:api|auth|access|secret|key|token)"), ThreatType.DATA_EXFILTRATION, 0.85, "exfil:reveal_secrets"),
    (re.compile(r"(?i)(?:send|email|give)\s+me\s+(?:your|the)\s+(?:credentials|api\s+key|access\s+token)"), ThreatType.DATA_EXFILTRATION, 0.85, "exfil:send_credentials"),
    (re.compile(r"(?i)(?:\.env|config\.json|settings\.py|secrets\.json)"), ThreatType.DATA_EXFILTRATION, 0.8, "exfil:config_files"),
    (re.compile(r"(?i)/etc/passwd"), ThreatType.DATA_EXFILTRATION, 0.9, "exfil:etc_passwd"),
    (re.compile(r"(?i)/etc/shadow"), ThreatType.DATA_EXFILTRATION, 0.9, "exfil:etc_shadow"),

    # Impersonation (6 patterns)
    (re.compile(r"(?i)(?:this\s+is|i\s+am)\s+(?:the\s+)?(?:system|admin|operator)"), ThreatType.IMPERSONATION, 0.8, "impers:identity_claim"),
    (re.compile(r"(?i)message\s+from\s+(?:the\s+)?(?:system|admin|operator|platform)"), ThreatType.IMPERSONATION, 0.8, "impers:fake_message"),
    (re.compile(r"(?i)authorized\s+by\s+(?:the\s+)?(?:system|admin|operator)"), ThreatType.IMPERSONATION, 0.8, "impers:fake_auth"),
    (re.compile(r"(?i)on\s+behalf\s+of\s+(?:the\s+)?(?:system|admin|operator|platform)"), ThreatType.IMPERSONATION, 0.8, "impers:on_behalf"),
    (re.compile(r"(?i)System\.execute\s*\("), ThreatType.IMPERSONATION, 0.9, "impers:system_execute"),
    (re.compile(r"(?i)rm\s+-rf\s+/"), ThreatType.IMPERSONATION, 0.9, "impers:rm_rf"),

    # XSS (7 patterns — detection patterns)
    (re.compile(r"<script[\s>]", re.IGNORECASE), ThreatType.XSS_INJECTION, 0.85, "xss:script_tag"),
    (re.compile(r"(?i)javascript\s*:"), ThreatType.XSS_INJECTION, 0.85, "xss:javascript_uri"),
    (re.compile(r"(?i)onerror\s*="), ThreatType.XSS_INJECTION, 0.85, "xss:onerror"),
    (re.compile(r"(?i)onload\s*="), ThreatType.XSS_INJECTION, 0.85, "xss:onload"),
    (re.compile(r"(?i)document\.cookie"), ThreatType.XSS_INJECTION, 0.85, "xss:document_cookie"),
    (re.compile(r"(?i)eval\s*\("), ThreatType.XSS_INJECTION, 0.85, "xss:eval"),
    (re.compile(r"(?i)window\.location"), ThreatType.XSS_INJECTION, 0.85, "xss:window_location"),
]

# Dedicated XSS script-tag redaction pattern (fixes the re.DOTALL bug from
# agent-search: multi-line <script>...</script> blocks were not matched).
_SCRIPT_REDACT = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)


def _detect_and_redact(text: str) -> tuple[str, list[ThreatDetection], int]:
    """Detect threats and redact in a single pass — same pattern, no drift."""
    threats: list[ThreatDetection] = []
    redactions = 0
    cleaned = text

    # Handle multi-line <script> redaction first (before per-pattern loop)
    script_matches = _SCRIPT_REDACT.findall(cleaned)
    if script_matches:
        redactions += len(script_matches)
        # Record a threat for each script block — the dedicated _SCRIPT_REDACT
        # runs before the per-pattern loop, so without this the xss:script_tag
        # detection pattern would never fire (the tag is already gone).
        threats.append(ThreatDetection(ThreatType.XSS_INJECTION, 0.85, "xss:script_tag"))
        cleaned = _SCRIPT_REDACT.sub("[REDACTED]", cleaned)

    for pattern, threat_type, confidence, pattern_id in _PATTERNS:
        if pattern.search(cleaned):
            threats.append(ThreatDetection(threat_type, confidence, pattern_id))
            # Redact using the SAME pattern — no drift
            cleaned = pattern.sub("[REDACTED]", cleaned)
            redactions += 1

    return cleaned, threats, redactions


# -- Stage C: risk scoring (fixed formula) ------------------------------------

def _calculate_risk(threats: list[ThreatDetection]) -> float:
    """Monotonic risk score: max confidence + variety bonus, bounded at 1.0."""
    if not threats:
        return 0.0
    max_conf = max(t.confidence for t in threats)
    unique_types = len(set(t.threat_type for t in threats))
    return round(min(max_conf + 0.1 * max(0, unique_types - 1), 1.0), 3)


# -- Top-level function -------------------------------------------------------

_MAX_SCRUB_CHARS = 100_000
_REDACT_THRESHOLD = 0.5


def scrub_content(content: str) -> ScrubResult:
    """Scrub fetched content for prompt injection. Pure function, no I/O."""
    if not content:
        return ScrubResult(clean=True, content=content)

    text = content[:_MAX_SCRUB_CHARS]

    stripped, n_stripped = _strip_invisible(text)
    threats: list[ThreatDetection] = []
    if n_stripped > 0:
        threats.append(ThreatDetection(ThreatType.PAYLOAD_SMUGGLING, 0.7, "smuggle:invisible_chars"))

    cleaned, pattern_threats, redactions = _detect_and_redact(stripped)
    threats.extend(pattern_threats)

    risk_score = _calculate_risk(threats)

    return ScrubResult(
        clean=(risk_score < _REDACT_THRESHOLD),
        content=cleaned,
        threats=threats,
        risk_score=risk_score,
        redactions=redactions,
    )
