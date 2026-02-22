"""VulnPatternAnalyzer — two-pass vulnerability pattern detection.

Pass 1: Deterministic regex/set checks — fast, testable, catches obvious patterns.
Pass 2: LLM batched analysis — semantic variants, chained vulnerabilities,
        false positive suppression, confidence refinement, context-specific descriptions.

Runs after Classifier, before Prioritizer.
"""

from __future__ import annotations

import logging
import re
import uuid
from urllib.parse import urlparse

from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.models import (
    AgentResult,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)

logger = logging.getLogger(__name__)

LLM_VULN_BATCH_SIZE = 12
# caps for LLM context windows — keep prompts under token limits
_MAX_SUMMARY_ENDPOINTS = 60
_MAX_RESPONSE_FIELDS_PER_ENDPOINT = 10

# severity ordering for comparison — lower index = higher severity
_SEVERITY_ORDER = {level: idx for idx, level in enumerate(RiskLevel)}

VULN_ANALYSIS_SYSTEM_PROMPT = """\
You are a principal application security researcher performing deep vulnerability analysis \
on web application endpoints.

CONTEXT:
You are the second pass of a two-pass vulnerability analyzer. Pass 1 (deterministic regex/set \
matching) has already run and produced initial indicators. Your job is to refine, enhance, and \
extend those results using semantic reasoning about the full application context.

ROLE:
Senior application security engineer specializing in API security, OWASP Top 10, and chained \
attack vectors. You understand how individual vulnerabilities compound when combined.

ACTION — Perform these 5 analyses on each endpoint batch:

1. SEMANTIC VARIANT DETECTION
   Identify vulnerability-relevant fields/params that regex missed. Examples:
   - Mass assignment: account_tier, is_premium, subscription_level, membership_type, plan, \
credit_limit, discount_rate, referral_bonus
   - IDOR: resource_id, ref, reference_number, slug, uuid, code, token (when used as lookup)
   - SSRF: endpoint, service_url, api_endpoint, forward_to, load_url, import_url
   Report these as new_indicators with the appropriate pattern type.

2. CHAINED VULNERABILITY IDENTIFICATION
   Look for endpoints where multiple vulnerability patterns combine into a higher-severity \
finding. Common chains:
   - auth_boundary_gap + bola_idor = unauthenticated IDOR (CRITICAL)
   - auth_boundary_gap + ssrf = unauthenticated SSRF (CRITICAL)
   - auth_boundary_gap + excessive_data_exposure = unauthenticated data leak (CRITICAL)
   - broken_function_level_auth + info_disclosure = admin data leak (CRITICAL)
   - file_upload + no auth = unauthenticated file upload (HIGH)
   Set chain_with to the endpoint key of related indicators.

3. FALSE POSITIVE SUPPRESSION
   Identify indicators that are likely false positives:
   - info_disclosure on /swagger or /api-docs when an OpenAPI spec was already parsed as a finding
   - BOLA on endpoints that are clearly public resources (e.g., /api/products/{id})
   - Generic low-confidence mass_assignment on standard registration endpoints with no \
dangerous fields
   Provide a clear reason for each suppression.

4. CONFIDENCE REFINEMENT
   Adjust confidence levels based on full context:
   - Upgrade: BOLA on a financial endpoint (e.g., /transfer/{account_number}) → CRITICAL
   - Upgrade: SSRF on an unauthenticated endpoint → CRITICAL
   - Downgrade: race_condition on an idempotent GET endpoint → LOW
   - Downgrade: info_disclosure on a path that returned 403 → INFO

5. CONTEXT-SPECIFIC DESCRIPTIONS
   For new or refined indicators, write descriptions that reference the actual URL, \
parameter names, tech stack, and business context rather than generic boilerplate.

FORMAT:
Respond with JSON:
{
  "reasoning": "2-3 sentences of chain-of-thought about patterns in this batch",
  "endpoint_analyses": [
    {
      "endpoint_key": "METHOD URL",
      "new_indicators": [
        {
          "pattern": "mass_assignment|bola_idor|ssrf|chained_vulnerability|...",
          "confidence": "critical|high|medium|low|info",
          "evidence": "specific evidence from the endpoint",
          "description": "context-specific description",
          "chain_with": "METHOD URL of related endpoint or null"
        }
      ],
      "suppressions": [
        {
          "original_pattern": "info_disclosure|...",
          "reason": "why this is a false positive"
        }
      ],
      "confidence_adjustments": [
        {
          "original_pattern": "bola_idor|...",
          "new_confidence": "critical|high|medium|low|info",
          "reason": "why confidence changed"
        }
      ]
    }
  ]
}

TONE:
Be precise and conservative. Only suppress indicators with clear justification. Only create \
new indicators when there is concrete evidence. Chained vulnerabilities must reference specific \
endpoint combinations.
"""

# ---------------------------------------------------------------------------
# Pattern constants — compiled regexes and keyword sets
# ---------------------------------------------------------------------------

# BOLA/IDOR: path parameters referencing object identifiers
_IDOR_PATH_PARAM = re.compile(
    r"\{(?:user_?id|account_?(?:id|number)|card_?id|order_?id|"
    r"transaction_?id|profile_?id|customer_?id|id|record_?id|"
    r"doc_?id|file_?id|message_?id|invoice_?id)\}",
    re.IGNORECASE,
)
_IDOR_QUERY_PARAMS = {
    "user_id", "userid", "account_id", "accountid", "account_number",
    "card_id", "order_id", "transaction_id", "profile_id", "id",
    "customer_id", "doc_id", "file_id", "record_id",
}

# Mass assignment: dangerous fields that shouldn't be user-controllable
_MASS_ASSIGN_FIELDS = {
    "role", "is_admin", "isadmin", "is_staff", "isstaff", "admin",
    "balance", "credit", "permissions", "privilege", "group",
    "verified", "is_verified", "is_active", "is_superuser",
    "account_type", "tier",
}
_MASS_ASSIGN_PATHS = re.compile(
    r"(?:/register|/signup|/profile|/user|/account|/settings)", re.IGNORECASE
)

# SSRF: params that accept URLs
_SSRF_PARAMS = {
    "url", "uri", "callback", "webhook", "redirect", "redirect_url",
    "redirect_uri", "image_url", "img_url", "icon_url", "avatar_url",
    "feed", "rss", "source", "target_url", "dest", "destination",
    "proxy", "fetch", "site", "link",
}

# File upload: content types and param names
_UPLOAD_PARAMS = {
    "file", "upload", "attachment", "document", "image", "photo",
    "avatar", "picture", "media",
}

# Race condition: financial/state-changing operations
_RACE_PATHS = re.compile(
    r"(?:/transfer|/payment|/withdraw|/deposit|/checkout|/redeem|"
    r"/purchase|/send|/claim|/apply|/vote|/confirm)",
    re.IGNORECASE,
)

# Prompt injection: AI-related paths and params
_AI_PATHS = re.compile(
    r"(?:/ai/|/chat|/bot|/assistant|/llm|/gpt|/copilot|/generate|/complete|/predict)",
    re.IGNORECASE,
)
_AI_PARAMS = {
    "message", "prompt", "query", "input", "text", "question",
    "instruction", "system_prompt", "context",
}

# Info disclosure: sensitive paths
_INFO_DISCLOSURE_PATHS = re.compile(
    r"(?:/debug|/config|/internal|/console|/actuator|/trace|/env|"
    r"/phpinfo|/server-status|/server-info|/elmah|/.env|"
    r"/secret|/backup|/dump|/swagger|/api-docs|/s3cr3t|/hidden)",
    re.IGNORECASE,
)

# Excessive data exposure: sensitive fields in response schemas
_SENSITIVE_RESPONSE_FIELDS = {
    "password", "passwd", "pass", "secret", "ssn", "social_security",
    "cvv", "cvc", "credit_card", "card_number", "cc_number",
    "token", "api_key", "apikey", "private_key", "secret_key",
    "pin", "tax_id",
}

# Numeric path segments longer than this are likely not enumerable IDs
_MAX_NUMERIC_ID_LENGTH = 10

# API version confusion: extract version from path
_VERSION_PATTERN = re.compile(r"/(?:api/)?v(\d+)(/.*)")

# Admin/debug paths for broken function-level auth
_ADMIN_DEBUG_PATHS = re.compile(
    r"(?:/admin|/debug|/internal|/management|/actuator|/console|/s3cr3t|/secret)",
    re.IGNORECASE,
)


def _apply_suppressions(
    indicators: list[VulnIndicator], suppressions: list[dict]
) -> None:
    """Mark matching indicators as suppressed based on LLM output."""
    for suppression in suppressions:
        pattern_val = suppression.get("original_pattern", "")
        reason = suppression.get("reason", "")
        for ind in indicators:
            if ind.pattern.value == pattern_val and not ind.suppressed:
                ind.suppressed = True
                ind.description += f" [SUPPRESSED by LLM: {reason}]"


def _apply_confidence_adjustments(
    indicators: list[VulnIndicator], adjustments: list[dict]
) -> None:
    """Update confidence levels on existing indicators based on LLM output."""
    for adj in adjustments:
        pattern_val = adj.get("original_pattern", "")
        new_conf = adj.get("new_confidence", "")
        reason = adj.get("reason", "")
        for ind in indicators:
            if ind.pattern.value != pattern_val or ind.suppressed:
                continue
            try:
                ind.confidence = RiskLevel(new_conf)
            except ValueError:
                continue
            ind.llm_enhanced = True
            ind.description += f" [Confidence adjusted by LLM: {reason}]"


def _apply_new_indicators(
    state: ScanState,
    existing: list[VulnIndicator],
    ep_key: str,
    new_indicators: list[dict],
) -> int:
    """Create new VulnIndicators from LLM output and link chains. Returns count added."""
    added = 0
    chain_id: str | None = None

    for raw in new_indicators:
        try:
            pattern = VulnPattern(raw.get("pattern", ""))
        except ValueError:
            continue
        try:
            confidence = RiskLevel(raw.get("confidence", "medium"))
        except ValueError:
            confidence = RiskLevel.MEDIUM

        chain_with = raw.get("chain_with")
        if chain_with or pattern == VulnPattern.CHAINED_VULNERABILITY:
            if chain_id is None:
                chain_id = uuid.uuid4().hex[:8]

        indicator = VulnIndicator(
            pattern=pattern,
            confidence=confidence,
            evidence=raw.get("evidence", ""),
            description=raw.get("description", ""),
            llm_enhanced=True,
            chain_id=chain_id,
        )
        existing.append(indicator)
        added += 1

        if chain_with and chain_with in state.vuln_indicators:
            _link_chain_to_related(state.vuln_indicators[chain_with], chain_id)

    return added


def _link_chain_to_related(
    related_indicators: list[VulnIndicator], chain_id: str | None
) -> None:
    """Tag the first un-chained, non-suppressed indicator with the given chain_id."""
    for ind in related_indicators:
        if ind.chain_id is None and not ind.suppressed:
            ind.chain_id = chain_id
            break


def _collect_param_names(ep: Endpoint) -> set[str]:
    """Gather all parameter names from an endpoint (params + details + body fields), lowercased."""
    names = {p.lower() for p in ep.parameters}
    names.update(pd.name.lower() for pd in ep.parameter_details)
    names.update(f.lower() for f in ep.request_body_fields)
    return names


@register_agent
class VulnPatternAnalyzer(BaseAgent):
    name = "vuln_analyzer"
    description = (
        "Analyzes discovered endpoints for vulnerability patterns using structural "
        "analysis of paths, parameters, schemas, and auth boundaries."
    )

    _PER_ENDPOINT_CHECKS = [
        "_check_bola_idor",
        "_check_mass_assignment",
        "_check_ssrf",
        "_check_file_upload",
        "_check_jwt_weakness",
        "_check_race_condition",
        "_check_prompt_injection",
        "_check_info_disclosure",
        "_check_auth_boundary",
        "_check_excessive_data",
    ]

    async def run(self, state: ScanState) -> AgentResult:
        all_endpoints = list(state.endpoints.items())
        if not all_endpoints:
            return AgentResult(agent_name=self.name, success=True)

        logger.info("Pass 1: deterministic pattern analysis on %d endpoints", len(all_endpoints))
        pass1_count = self._run_per_endpoint_checks(state, all_endpoints)
        pass1_count += self._merge_cross_endpoint(state, self._check_version_confusion(all_endpoints))
        pass1_count += self._merge_cross_endpoint(state, self._check_broken_function_auth(all_endpoints))
        logger.info("Pass 1 complete: %d indicators found", pass1_count)

        logger.info("Pass 2: LLM analysis for semantic variants, chains, and refinement")
        pass2_count = await self._llm_analysis_pass(state)
        logger.info("Pass 2 complete: %d additional/modified indicators", pass2_count)

        findings = self._build_findings(state)
        return AgentResult(
            agent_name=self.name,
            success=True,
            endpoints_found=[],
            findings=findings,
        )

    def _run_per_endpoint_checks(
        self, state: ScanState, all_endpoints: list[tuple[str, Endpoint]]
    ) -> int:
        total = 0
        for key, ep in all_endpoints:
            indicators: list[VulnIndicator] = []
            for method_name in self._PER_ENDPOINT_CHECKS:
                indicators.extend(getattr(self, method_name)(ep))
            if indicators:
                state.vuln_indicators[key] = indicators
                total += len(indicators)
        return total

    @staticmethod
    def _merge_cross_endpoint(
        state: ScanState, results: dict[str, list[VulnIndicator]]
    ) -> int:
        total = 0
        for key, inds in results.items():
            existing = state.vuln_indicators.get(key, [])
            existing.extend(inds)
            state.vuln_indicators[key] = existing
            total += len(inds)
        return total

    def _build_findings(self, state: ScanState) -> list[Finding]:
        findings: list[Finding] = []
        active_count, suppressed_count = self._count_indicators(state)

        pattern_counts = self._count_patterns(state)
        if not pattern_counts:
            return findings

        dist = ", ".join(
            f"{v} {k}" for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])
        )
        suppressed_note = f" ({suppressed_count} suppressed by LLM)" if suppressed_count else ""
        findings.append(Finding(
            agent_name=self.name,
            finding_type="vuln_pattern_analysis",
            title=(
                f"Identified {active_count} vulnerability indicators "
                f"across {len(state.vuln_indicators)} endpoints{suppressed_note}"
            ),
            detail=f"Pattern distribution: {dist}",
            severity=RiskLevel.INFO,
        ))

        # emit chain findings first, track which indicators are already reported
        chained_chain_ids = self._build_chain_findings(state, findings)

        # emit individual findings for high/critical non-chained indicators
        self._build_individual_findings(state, findings, chained_chain_ids)

        return findings

    @staticmethod
    def _count_indicators(state: ScanState) -> tuple[int, int]:
        all_inds = [ind for inds in state.vuln_indicators.values() for ind in inds]
        suppressed = sum(1 for ind in all_inds if ind.suppressed)
        return len(all_inds) - suppressed, suppressed

    @staticmethod
    def _count_patterns(state: ScanState) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ind in (i for inds in state.vuln_indicators.values() for i in inds if not i.suppressed):
            counts[ind.pattern.value] = counts.get(ind.pattern.value, 0) + 1
        return counts

    def _build_chain_findings(
        self, state: ScanState, findings: list[Finding]
    ) -> set[str]:
        """Emit grouped findings for chained vulnerabilities. Returns emitted chain_ids."""
        seen_chains: set[str] = set()
        for key, indicators in state.vuln_indicators.items():
            for ind in indicators:
                if ind.suppressed or not ind.chain_id:
                    continue
                if ind.chain_id in seen_chains:
                    continue
                seen_chains.add(ind.chain_id)

                chain_members = self._collect_chain_members(state, ind.chain_id)
                if len(chain_members) <= 1:
                    continue

                chain_patterns = " + ".join(i.pattern.value for _, i in chain_members)
                chain_endpoints = ", ".join(dict.fromkeys(k for k, _ in chain_members))
                severity = min(
                    (i.confidence for _, i in chain_members),
                    key=lambda r: _SEVERITY_ORDER[r],
                )
                findings.append(Finding(
                    agent_name=self.name,
                    finding_type="vuln_chained_vulnerability",
                    title=f"[LLM] CHAINED VULNERABILITY — {chain_patterns}",
                    detail=(
                        f"Combined attack chain across: {chain_endpoints}. "
                        + chain_members[0][1].description
                    ),
                    severity=severity,
                    evidence="; ".join(i.evidence for _, i in chain_members),
                ))

        return seen_chains

    @staticmethod
    def _collect_chain_members(
        state: ScanState, chain_id: str
    ) -> list[tuple[str, VulnIndicator]]:
        return [
            (k, i)
            for k, inds in state.vuln_indicators.items()
            for i in inds
            if i.chain_id == chain_id and not i.suppressed
        ]

    def _build_individual_findings(
        self,
        state: ScanState,
        findings: list[Finding],
        chained_chain_ids: set[str],
    ) -> None:
        """Emit individual findings for high/critical indicators not part of a chain."""
        for key, indicators in state.vuln_indicators.items():
            for ind in indicators:
                if ind.suppressed:
                    continue
                if ind.chain_id and ind.chain_id in chained_chain_ids:
                    continue
                if ind.confidence not in (RiskLevel.CRITICAL, RiskLevel.HIGH):
                    continue

                tag = "[LLM] " if ind.llm_enhanced else ""
                findings.append(Finding(
                    agent_name=self.name,
                    finding_type=f"vuln_{ind.pattern.value}",
                    title=f"{tag}{ind.pattern.value.upper()} — {key}",
                    detail=ind.description,
                    severity=ind.confidence,
                    evidence=ind.evidence,
                ))

    # ------------------------------------------------------------------
    # Per-endpoint pattern checkers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_bola_idor(ep: Endpoint) -> list[VulnIndicator]:
        indicators: list[VulnIndicator] = []
        path = urlparse(ep.url).path

        if _IDOR_PATH_PARAM.search(path):
            indicators.append(VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence=f"Path contains object ID template: {path}",
                description=(
                    "Endpoint accepts an object identifier in the URL path. "
                    "An attacker could enumerate or substitute IDs to access "
                    "other users' resources if authorization is insufficient."
                ),
            ))

        for part in path.rstrip("/").split("/"):
            if part.isdigit() and len(part) <= _MAX_NUMERIC_ID_LENGTH:
                indicators.append(VulnIndicator(
                    pattern=VulnPattern.BOLA_IDOR,
                    confidence=RiskLevel.MEDIUM,
                    evidence=f"Numeric path segment in {path}",
                    description=(
                        "Numeric path segment detected — may be an enumerable "
                        "object identifier susceptible to IDOR."
                    ),
                ))
                break

        idor_params = _collect_param_names(ep) & _IDOR_QUERY_PARAMS
        if idor_params:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence=f"ID-like params: {', '.join(sorted(idor_params))}",
                description=(
                    "Parameters reference object identifiers that could be "
                    "manipulated for unauthorized access."
                ),
            ))

        return indicators

    @staticmethod
    def _check_mass_assignment(ep: Endpoint) -> list[VulnIndicator]:
        if ep.method not in ("POST", "PUT", "PATCH"):
            return []
        path = urlparse(ep.url).path
        if not _MASS_ASSIGN_PATHS.search(path):
            return []

        dangerous = _collect_param_names(ep) & _MASS_ASSIGN_FIELDS
        if dangerous:
            return [VulnIndicator(
                pattern=VulnPattern.MASS_ASSIGNMENT,
                confidence=RiskLevel.HIGH,
                evidence=f"Dangerous fields accepted: {', '.join(sorted(dangerous))}",
                description=(
                    "Endpoint accepts privilege-related fields that could allow "
                    "an attacker to escalate privileges via mass assignment."
                ),
            )]

        # even without known dangerous fields, POST to register/profile is worth flagging
        return [VulnIndicator(
            pattern=VulnPattern.MASS_ASSIGNMENT,
            confidence=RiskLevel.LOW,
            evidence=f"{ep.method} {path}",
            description=(
                "State-changing request to a user-facing endpoint. Verify that "
                "the server restricts which fields can be set by the client."
            ),
        )]

    @staticmethod
    def _check_ssrf(ep: Endpoint) -> list[VulnIndicator]:
        ssrf_params = _collect_param_names(ep) & _SSRF_PARAMS
        if ssrf_params:
            return [VulnIndicator(
                pattern=VulnPattern.SSRF,
                confidence=RiskLevel.HIGH,
                evidence=f"URL-accepting params: {', '.join(sorted(ssrf_params))}",
                description=(
                    "Endpoint accepts URL or URI parameters that could be exploited "
                    "for Server-Side Request Forgery to reach internal services."
                ),
            )]

        # fall back to path-based detection
        path = urlparse(ep.url).path.lower()
        if "url" not in path and "callback" not in path and "webhook" not in path:
            return []

        return [VulnIndicator(
            pattern=VulnPattern.SSRF,
            confidence=RiskLevel.MEDIUM,
            evidence=f"URL-related path: {path}",
            description=(
                "Endpoint path suggests it accepts URLs. Verify whether user input "
                "could trigger server-side requests to internal services."
            ),
        )]

    @staticmethod
    def _check_file_upload(ep: Endpoint) -> list[VulnIndicator]:
        indicators: list[VulnIndicator] = []

        if ep.request_body_content_type and "multipart" in ep.request_body_content_type:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.FILE_UPLOAD,
                confidence=RiskLevel.HIGH,
                evidence=f"Content-Type: {ep.request_body_content_type}",
                description=(
                    "Endpoint accepts multipart file uploads. Verify file type "
                    "validation, size limits, and storage isolation."
                ),
            ))

        upload_params = _collect_param_names(ep) & _UPLOAD_PARAMS
        path = urlparse(ep.url).path.lower()
        if upload_params:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.FILE_UPLOAD,
                confidence=RiskLevel.HIGH,
                evidence=f"Upload params: {', '.join(sorted(upload_params))}",
                description=(
                    "Endpoint appears to handle file uploads. Check for "
                    "unrestricted file types and path traversal."
                ),
            ))
        elif "upload" in path or "attach" in path:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.FILE_UPLOAD,
                confidence=RiskLevel.MEDIUM,
                evidence=f"Upload path: {path}",
                description=(
                    "Endpoint path suggests file upload functionality. Check for "
                    "unrestricted file types and path traversal."
                ),
            ))

        return indicators

    @staticmethod
    def _check_jwt_weakness(ep: Endpoint) -> list[VulnIndicator]:
        indicators: list[VulnIndicator] = []

        for scheme in ep.security_schemes:
            # JWT bearer
            if scheme.scheme_type == "http" and scheme.bearer_format.upper() == "JWT":
                indicators.append(VulnIndicator(
                    pattern=VulnPattern.JWT_WEAKNESS,
                    confidence=RiskLevel.MEDIUM,
                    evidence=f"Security scheme: {scheme.scheme_name} (JWT bearer)",
                    description=(
                        "JWT authentication detected. Test for algorithm confusion "
                        "(none/HS256 vs RS256), weak signing keys, and token expiry."
                    ),
                ))

            # API key in query string
            if scheme.scheme_type == "apiKey" and scheme.location == "query":
                indicators.append(VulnIndicator(
                    pattern=VulnPattern.JWT_WEAKNESS,
                    confidence=RiskLevel.HIGH,
                    evidence=f"API key in query: {scheme.scheme_name}",
                    description=(
                        "API key transmitted in query string — visible in logs, "
                        "browser history, and referrer headers."
                    ),
                ))

        # auth over HTTP (not HTTPS)
        if ep.url.startswith("http://") and ep.requires_auth:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.JWT_WEAKNESS,
                confidence=RiskLevel.HIGH,
                evidence=f"Auth required over HTTP: {ep.url}",
                description=(
                    "Authenticated endpoint served over unencrypted HTTP. "
                    "Credentials and tokens are exposed to network interception."
                ),
            ))

        return indicators

    @staticmethod
    def _check_race_condition(ep: Endpoint) -> list[VulnIndicator]:
        if ep.method != "POST":
            return []
        path = urlparse(ep.url).path
        if not _RACE_PATHS.search(path):
            return []
        return [VulnIndicator(
            pattern=VulnPattern.RACE_CONDITION,
            confidence=RiskLevel.HIGH,
            evidence=f"POST {path}",
            description=(
                "State-changing financial or transactional endpoint. "
                "Concurrent requests may exploit race conditions for "
                "duplicate transactions or balance manipulation."
            ),
        )]

    @staticmethod
    def _check_prompt_injection(ep: Endpoint) -> list[VulnIndicator]:
        path = urlparse(ep.url).path
        if not _AI_PATHS.search(path):
            return []

        ai_params = _collect_param_names(ep) & _AI_PARAMS
        evidence_parts = [f"AI path: {path}"]
        if ai_params:
            evidence_parts.append(f"input params: {', '.join(sorted(ai_params))}")

        return [VulnIndicator(
            pattern=VulnPattern.PROMPT_INJECTION,
            confidence=RiskLevel.HIGH if ai_params else RiskLevel.MEDIUM,
            evidence="; ".join(evidence_parts),
            description=(
                "AI/LLM-powered endpoint that accepts user input. "
                "Test for prompt injection to override system instructions, "
                "extract training data, or trigger unintended actions."
            ),
        )]

    @staticmethod
    def _check_info_disclosure(ep: Endpoint) -> list[VulnIndicator]:
        path = urlparse(ep.url).path
        if not _INFO_DISCLOSURE_PATHS.search(path):
            return []

        # only flag if the endpoint is actually reachable
        if ep.status_code and ep.status_code >= 400:
            return []

        return [VulnIndicator(
            pattern=VulnPattern.INFO_DISCLOSURE,
            confidence=RiskLevel.HIGH,
            evidence=f"Accessible sensitive path: {ep.method} {path} [{ep.status_code}]",
            description=(
                "Endpoint exposes internal configuration, debug, or administrative "
                "interface that may leak sensitive operational details."
            ),
        )]

    @staticmethod
    def _check_auth_boundary(ep: Endpoint) -> list[VulnIndicator]:
        if ep.requires_auth is not False:
            return []
        path = urlparse(ep.url).path.lower()
        data_patterns = (
            "/user", "/account", "/profile", "/transaction",
            "/balance", "/card", "/order", "/payment",
            "/message", "/notification",
        )
        if not any(seg in path for seg in data_patterns):
            return []

        return [VulnIndicator(
            pattern=VulnPattern.AUTH_BOUNDARY_GAP,
            confidence=RiskLevel.HIGH,
            evidence=f"No auth required: {ep.method} {path}",
            description=(
                "Data endpoint classified as not requiring authentication. "
                "Verify whether sensitive user data is accessible without credentials."
            ),
        )]

    @staticmethod
    def _check_excessive_data(ep: Endpoint) -> list[VulnIndicator]:
        sensitive = {f.lower() for f in ep.response_fields} & _SENSITIVE_RESPONSE_FIELDS
        if not sensitive:
            return []
        return [VulnIndicator(
            pattern=VulnPattern.EXCESSIVE_DATA_EXPOSURE,
            confidence=RiskLevel.HIGH,
            evidence=f"Sensitive response fields: {', '.join(sorted(sensitive))}",
            description=(
                "API response schema includes fields containing credentials or "
                "PII that should not be returned to clients."
            ),
        )]

    # ------------------------------------------------------------------
    # Pass 2: LLM analysis
    # ------------------------------------------------------------------

    async def _llm_analysis_pass(self, state: ScanState) -> int:
        """Orchestrate batched LLM calls for semantic vuln analysis."""
        all_endpoints = list(state.endpoints.items())
        if not all_endpoints:
            return 0

        cross_ref = self._format_all_endpoints_summary(state, all_endpoints)
        tech_context = self.build_tech_context(state)
        findings_context = self.build_findings_context(state)

        total_additions = 0
        for i in range(0, len(all_endpoints), LLM_VULN_BATCH_SIZE):
            batch = all_endpoints[i : i + LLM_VULN_BATCH_SIZE]
            try:
                batch_context = self._format_batch_for_llm(state, batch)
                messages = [
                    {"role": "system", "content": VULN_ANALYSIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Target: {state.target}\n\n"
                            f"Tech Stack:\n{tech_context}\n\n"
                            f"Prior Findings:\n{findings_context}\n\n"
                            f"All Endpoints Summary (for cross-referencing):\n{cross_ref}\n\n"
                            f"BATCH TO ANALYZE ({len(batch)} endpoints):\n{batch_context}"
                        ),
                    },
                ]

                data = await self.llm.chat_json(
                    messages, name=f"vuln_llm_batch_{i // LLM_VULN_BATCH_SIZE}"
                )
                total_additions += self._process_llm_vuln_results(state, data, batch)

            except Exception as e:
                logger.warning(
                    "LLM vuln analysis batch %d failed (rule-based results intact): %s",
                    i // LLM_VULN_BATCH_SIZE, e,
                )
                continue

        return total_additions

    @staticmethod
    def _format_batch_for_llm(
        state: ScanState, batch: list[tuple[str, Endpoint]]
    ) -> str:
        """Format endpoints and their rule-based indicators for LLM context."""
        lines = []
        for key, ep in batch:
            auth = (
                "requires_auth"
                if ep.requires_auth
                else "no_auth" if ep.requires_auth is False
                else "auth_unknown"
            )
            cat = ep.category.value if ep.category else "unknown"
            params = ", ".join(ep.parameters) if ep.parameters else "none"
            body_fields = ", ".join(ep.request_body_fields) if ep.request_body_fields else "none"
            resp_fields = (
                ", ".join(ep.response_fields[:_MAX_RESPONSE_FIELDS_PER_ENDPOINT])
                if ep.response_fields else "none"
            )

            lines.append(
                f"\n--- {key} ---\n"
                f"  Category: {cat} | Auth: {auth} | Status: {ep.status_code}\n"
                f"  Params: {params}\n"
                f"  Body fields: {body_fields}\n"
                f"  Response fields: {resp_fields}"
            )

            indicators = state.vuln_indicators.get(key, [])
            if indicators:
                ind_lines = [
                    f"    - {ind.pattern.value} ({ind.confidence.value}): {ind.evidence}"
                    for ind in indicators
                ]
                lines.append("  Rule-based indicators:\n" + "\n".join(ind_lines))
            else:
                lines.append("  Rule-based indicators: none")

        return "\n".join(lines)

    @staticmethod
    def _format_all_endpoints_summary(
        state: ScanState, all_endpoints: list[tuple[str, Endpoint]]
    ) -> str:
        """Short cross-reference summary of all endpoints."""
        lines = []
        for key, ep in all_endpoints[:_MAX_SUMMARY_ENDPOINTS]:
            auth = "auth" if ep.requires_auth else "noauth" if ep.requires_auth is False else "?"
            vuln_count = len(state.vuln_indicators.get(key, []))
            vuln_tag = f" [{vuln_count} vulns]" if vuln_count else ""
            lines.append(f"  {key} ({auth}){vuln_tag}")
        if len(all_endpoints) > _MAX_SUMMARY_ENDPOINTS:
            lines.append(f"  ... and {len(all_endpoints) - _MAX_SUMMARY_ENDPOINTS} more")
        return "\n".join(lines)

    @staticmethod
    def _process_llm_vuln_results(
        state: ScanState, data: dict, batch: list[tuple[str, Endpoint]]
    ) -> int:
        """Merge LLM output back into state vuln_indicators."""
        total = 0
        batch_keys = {key for key, _ in batch}

        for analysis in data.get("endpoint_analyses", []):
            ep_key = analysis.get("endpoint_key", "")
            if ep_key not in batch_keys:
                continue

            existing = state.vuln_indicators.get(ep_key, [])
            _apply_suppressions(indicators=existing, suppressions=analysis.get("suppressions", []))
            _apply_confidence_adjustments(
                indicators=existing, adjustments=analysis.get("confidence_adjustments", [])
            )
            total += _apply_new_indicators(
                state=state,
                existing=existing,
                ep_key=ep_key,
                new_indicators=analysis.get("new_indicators", []),
            )
            state.vuln_indicators[ep_key] = existing

        return total

    # ------------------------------------------------------------------
    # Cross-endpoint analyzers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_version_confusion(
        all_endpoints: list[tuple[str, Endpoint]],
    ) -> dict[str, list[VulnIndicator]]:
        """Detect same resource across API versions with inconsistent auth."""
        # group endpoints by version-stripped resource path
        resources: dict[str, list[tuple[str, str, Endpoint]]] = {}
        for key, ep in all_endpoints:
            path = urlparse(ep.url).path
            m = _VERSION_PATTERN.search(path)
            if not m:
                continue
            version, resource = m.group(1), m.group(2)
            resources.setdefault(resource, []).append((key, version, ep))

        results: dict[str, list[VulnIndicator]] = {}
        for resource, versions in resources.items():
            if len(versions) < 2:
                continue
            auth_set = {(v, ep.requires_auth) for _, v, ep in versions}
            auth_values = {a for _, a in auth_set}
            if len(auth_values) > 1 and None not in auth_values:
                for key, ver, ep in versions:
                    results.setdefault(key, []).append(VulnIndicator(
                        pattern=VulnPattern.API_VERSION_CONFUSION,
                        confidence=RiskLevel.HIGH,
                        evidence=(
                            f"Resource {resource} — v{ver} auth={ep.requires_auth}, "
                            f"other versions differ"
                        ),
                        description=(
                            "Same resource exists across API versions with inconsistent "
                            "authentication requirements. Older versions may lack "
                            "security controls added in newer versions."
                        ),
                    ))
        return results

    @staticmethod
    def _check_broken_function_auth(
        all_endpoints: list[tuple[str, Endpoint]],
    ) -> dict[str, list[VulnIndicator]]:
        """Detect admin/debug endpoints accessible without auth."""
        results: dict[str, list[VulnIndicator]] = {}
        for key, ep in all_endpoints:
            path = urlparse(ep.url).path
            if not _ADMIN_DEBUG_PATHS.search(path):
                continue
            if ep.requires_auth is not False:
                continue
            if ep.status_code and ep.status_code >= 400:
                continue
            results.setdefault(key, []).append(VulnIndicator(
                pattern=VulnPattern.BROKEN_FUNCTION_LEVEL_AUTH,
                confidence=RiskLevel.CRITICAL,
                evidence=f"Admin/debug path without auth: {ep.method} {path} [{ep.status_code}]",
                description=(
                    "Administrative or debug endpoint is accessible without "
                    "authentication, potentially allowing privilege escalation "
                    "or sensitive data access."
                ),
            ))
        return results
