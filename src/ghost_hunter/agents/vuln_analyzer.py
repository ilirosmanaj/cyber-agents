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
from src.ghost_hunter.config import settings
from src.ghost_hunter.models import (
    AgentResult,
    Endpoint,
    Finding,
    RiskLevel,
    ScanState,
    VulnIndicator,
    VulnPattern,
)
from src.ghost_hunter.models.llm_responses import VulnAnalysisBatchResponse

logger = logging.getLogger(__name__)

LLM_VULN_BATCH_SIZE = 8
# caps for LLM context windows — keep prompts under token limits
_MAX_SUMMARY_ENDPOINTS = 100
_MAX_RESPONSE_FIELDS_PER_ENDPOINT = 25

# severity ordering for comparison — lower index = higher severity
_SEVERITY_ORDER = {level: idx for idx, level in enumerate(RiskLevel)}


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

# Response body leak patterns
_INTERNAL_IP_PATTERN = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
_STACK_TRACE_PATTERN = re.compile(
    r"(?:Traceback \(most recent|at (?:com|org|net|io)\.\w+|"
    r"Exception in thread|java\.\w+Exception|System\.NullReferenceException|"
    r"File \"[^\"]+\", line \d+)",
    re.IGNORECASE,
)
_SQL_FRAGMENT_PATTERN = re.compile(
    r"(?:SQL(?:State|Exception)|mysql_|pg_|ORA-\d+|"
    r"(?:syntax error|You have an error in your SQL|Unclosed quotation mark))",
    re.IGNORECASE,
)
_API_KEY_TOKEN_PATTERN = re.compile(
    r"(?:sk-[a-zA-Z0-9]{20,}|pk_(?:live|test)_[a-zA-Z0-9]{20,}|"
    r"ghp_[a-zA-Z0-9]{36}|eyJ[a-zA-Z0-9_-]{20,}\.eyJ)",
)
_EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
)

# more than this many emails in a single response suggests a data dump
_EXCESSIVE_EMAIL_THRESHOLD = 3

# Numeric path segments longer than this are likely not enumerable IDs
_MAX_NUMERIC_ID_LENGTH = 10

# version prefix — segments like "v1", "v2" should not trigger BOLA
_VERSION_SEGMENT = re.compile(r"^v\d+$", re.IGNORECASE)

# API version confusion: extract version from path
_VERSION_PATTERN = re.compile(r"/(?:api/)?v(\d+)(/.*)")

# Admin/debug paths for broken function-level auth
_ADMIN_DEBUG_PATHS = re.compile(
    r"(?:/admin|/debug|/internal|/management|/actuator|/console|/s3cr3t|/secret)",
    re.IGNORECASE,
)

# SQL injection surface: params that likely build SQL
_SQL_INJECTION_PARAMS = {
    "q", "query", "search", "filter", "order_by", "sort",
    "sort_by", "order", "group_by", "where", "column",
    "field", "table", "select", "limit", "offset",
}

# path traversal surface: params that reference files
_PATH_TRAVERSAL_PARAMS = {
    "file", "path", "document", "filename", "filepath",
    "dir", "directory", "folder", "template", "include",
    "page", "read", "load", "config", "log",
}

# command injection surface: params that suggest OS interaction
_COMMAND_INJECTION_PARAMS = {
    "cmd", "exec", "command", "execute", "run",
    "shell", "process", "ping", "host", "ip",
}

# SSRF — segment-level path matching (avoids false positives from substrings)
_SSRF_PATH_SEGMENTS = {"url", "callback", "webhook", "proxy", "fetch"}


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
        "_check_response_body_leaks",
        "_check_injection_surfaces",
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

        match = _IDOR_PATH_PARAM.search(path)
        if match:
            param = match.group(0).strip("{}")
            indicators.append(VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence=f"Path contains object ID template: {path}",
                description=(
                    f"Path parameter '{param}' in {path} is an enumerable object "
                    f"identifier. Changing this value may return another user's "
                    f"data if server-side authorization doesn't verify resource "
                    f"ownership."
                ),
            ))

        segments = path.rstrip("/").split("/")
        for i, part in enumerate(segments):
            if not part.isdigit() or len(part) > _MAX_NUMERIC_ID_LENGTH:
                continue
            # skip version-like segments (e.g., "1" after "v" in /api/v1/users)
            prev = segments[i - 1] if i > 0 else ""
            if _VERSION_SEGMENT.match(prev + part):
                continue
            indicators.append(VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.MEDIUM,
                evidence=f"Numeric path segment in {path}",
                description=(
                    f"Numeric segment '{part}' in {path} may be an enumerable "
                    f"object identifier. An attacker could iterate over IDs to "
                    f"access other records."
                ),
            ))
            break

        idor_params = _collect_param_names(ep) & _IDOR_QUERY_PARAMS
        if idor_params:
            param_list = ", ".join(sorted(idor_params))
            indicators.append(VulnIndicator(
                pattern=VulnPattern.BOLA_IDOR,
                confidence=RiskLevel.HIGH,
                evidence=f"ID-like params: {param_list}",
                description=(
                    f"Parameters {param_list} on {ep.method} {path} reference "
                    f"object identifiers that could be manipulated to access "
                    f"another user's resources."
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

        all_params = _collect_param_names(ep)
        dangerous = all_params & _MASS_ASSIGN_FIELDS
        if dangerous:
            field_list = ", ".join(sorted(dangerous))
            return [VulnIndicator(
                pattern=VulnPattern.MASS_ASSIGNMENT,
                confidence=RiskLevel.HIGH,
                evidence=f"Dangerous fields accepted: {field_list}",
                description=(
                    f"Privilege-related fields ({field_list}) are accepted "
                    f"by {ep.method} {path}. An attacker could include them "
                    f"in the request body to escalate privileges or modify "
                    f"account attributes they shouldn't control."
                ),
            )]

        # only emit LOW if the endpoint actually accepts user-controllable fields
        if not all_params:
            return []

        return [VulnIndicator(
            pattern=VulnPattern.MASS_ASSIGNMENT,
            confidence=RiskLevel.LOW,
            evidence=f"{ep.method} {path}",
            description=(
                f"{ep.method} {path} is a state-changing request to a "
                f"user-facing endpoint. Verify that the server restricts "
                f"which fields can be set by the client."
            ),
        )]

    @staticmethod
    def _check_ssrf(ep: Endpoint) -> list[VulnIndicator]:
        path = urlparse(ep.url).path
        ssrf_params = _collect_param_names(ep) & _SSRF_PARAMS
        if ssrf_params:
            param_list = ", ".join(sorted(ssrf_params))
            return [VulnIndicator(
                pattern=VulnPattern.SSRF,
                confidence=RiskLevel.HIGH,
                evidence=f"URL-accepting params: {param_list}",
                description=(
                    f"URL parameters ({param_list}) on {ep.method} {path} "
                    f"could be exploited for Server-Side Request Forgery. "
                    f"Supplying internal URLs (e.g., http://169.254.169.254/) "
                    f"may reach cloud metadata or internal services."
                ),
            )]

        # segment-level matching: check if any path segment starts with an SSRF keyword
        # handles hyphenated segments like "url-proxy" and "webhook-handler"
        path_segments = [seg.lower() for seg in path.split("/") if seg]
        has_ssrf_segment = any(
            seg in _SSRF_PATH_SEGMENTS or any(seg.startswith(kw + "-") or seg.startswith(kw + "_") for kw in _SSRF_PATH_SEGMENTS)
            for seg in path_segments
        )
        if not has_ssrf_segment:
            return []

        return [VulnIndicator(
            pattern=VulnPattern.SSRF,
            confidence=RiskLevel.MEDIUM,
            evidence=f"URL-related path: {path}",
            description=(
                f"{ep.method} {path} path suggests it processes URLs. "
                f"Verify whether user input could trigger server-side "
                f"requests to internal services."
            ),
        )]

    @staticmethod
    def _check_file_upload(ep: Endpoint) -> list[VulnIndicator]:
        indicators: list[VulnIndicator] = []
        path = urlparse(ep.url).path

        if ep.request_body_content_type and "multipart" in ep.request_body_content_type:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.FILE_UPLOAD,
                confidence=RiskLevel.HIGH,
                evidence=f"Content-Type: {ep.request_body_content_type}",
                description=(
                    f"Multipart uploads ({ep.request_body_content_type}) "
                    f"on {ep.method} {path}. Verify file type validation, "
                    f"size limits, and storage isolation."
                ),
            ))

        upload_params = _collect_param_names(ep) & _UPLOAD_PARAMS
        path_lower = path.lower()
        if upload_params:
            param_list = ", ".join(sorted(upload_params))
            indicators.append(VulnIndicator(
                pattern=VulnPattern.FILE_UPLOAD,
                confidence=RiskLevel.HIGH,
                evidence=f"Upload params: {param_list}",
                description=(
                    f"{ep.method} {path} handles file uploads via parameters "
                    f"({param_list}). Check for unrestricted file types and "
                    f"path traversal in filenames."
                ),
            ))
        elif "upload" in path_lower or "attach" in path_lower:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.FILE_UPLOAD,
                confidence=RiskLevel.MEDIUM,
                evidence=f"Upload path: {path}",
                description=(
                    f"{ep.method} {path} path suggests file upload functionality. "
                    f"Check for unrestricted file types and path traversal."
                ),
            ))

        return indicators

    @staticmethod
    def _check_jwt_weakness(ep: Endpoint) -> list[VulnIndicator]:
        indicators: list[VulnIndicator] = []
        path = urlparse(ep.url).path

        for scheme in ep.security_schemes:
            if scheme.scheme_type == "http" and scheme.bearer_format.upper() == "JWT":
                indicators.append(VulnIndicator(
                    pattern=VulnPattern.JWT_WEAKNESS,
                    confidence=RiskLevel.MEDIUM,
                    evidence=f"Security scheme: {scheme.scheme_name} (JWT bearer)",
                    description=(
                        f"JWT bearer auth (scheme: {scheme.scheme_name}) on "
                        f"{ep.method} {path}. Test for algorithm confusion "
                        f"(none/HS256 vs RS256), weak signing keys, and "
                        f"missing token expiry."
                    ),
                ))

            if scheme.scheme_type == "apiKey" and scheme.location == "query":
                indicators.append(VulnIndicator(
                    pattern=VulnPattern.JWT_WEAKNESS,
                    confidence=RiskLevel.HIGH,
                    evidence=f"API key in query: {scheme.scheme_name}",
                    description=(
                        f"{ep.method} {path} transmits API key "
                        f"'{scheme.scheme_name}' in query string — visible "
                        f"in server logs, browser history, and referrer headers."
                    ),
                ))

        if ep.url.startswith("http://") and ep.requires_auth:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.JWT_WEAKNESS,
                confidence=RiskLevel.HIGH,
                evidence=f"Auth required over HTTP: {ep.url}",
                description=(
                    f"{ep.method} {path} requires authentication but is "
                    f"served over unencrypted HTTP. Credentials and tokens "
                    f"are exposed to network interception."
                ),
            ))

        # detect JWT/Bearer from response headers or notes
        if not indicators:
            auth_header = ep.response_headers.get("www-authenticate", "")
            notes_lower = ep.notes.lower()
            if "bearer" in auth_header.lower() or "jwt" in notes_lower:
                indicators.append(VulnIndicator(
                    pattern=VulnPattern.JWT_WEAKNESS,
                    confidence=RiskLevel.MEDIUM,
                    evidence=f"Bearer/JWT detected in headers or notes: {auth_header or ep.notes[:60]}",
                    description=(
                        f"{ep.method} {path} uses JWT/Bearer auth (detected "
                        f"from response headers). Test for algorithm confusion "
                        f"and weak signing keys."
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
                f"Transactional endpoint at POST {path}. Concurrent "
                f"requests may exploit race conditions to duplicate "
                f"transactions or manipulate balances."
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
            input_params = ", ".join(sorted(ai_params))
            evidence_parts.append(f"input params: {input_params}")
            desc = (
                f"AI/LLM endpoint at {path} accepts free-text input via "
                f"'{input_params}'. An attacker could inject prompts to "
                f"override system instructions, exfiltrate the system "
                f"prompt, or trigger unintended tool calls."
            )
        else:
            desc = (
                f"AI/LLM endpoint at {path} may accept user input. "
                f"Test for prompt injection to override system instructions, "
                f"extract training data, or trigger unintended actions."
            )

        return [VulnIndicator(
            pattern=VulnPattern.PROMPT_INJECTION,
            confidence=RiskLevel.HIGH if ai_params else RiskLevel.MEDIUM,
            evidence="; ".join(evidence_parts),
            description=desc,
        )]

    @staticmethod
    def _check_info_disclosure(ep: Endpoint) -> list[VulnIndicator]:
        path = urlparse(ep.url).path
        if not _INFO_DISCLOSURE_PATHS.search(path):
            return []

        if ep.status_code and ep.status_code >= 400:
            return []

        status = f" [{ep.status_code}]" if ep.status_code else ""
        return [VulnIndicator(
            pattern=VulnPattern.INFO_DISCLOSURE,
            confidence=RiskLevel.HIGH,
            evidence=f"Accessible sensitive path: {ep.method} {path}{status}",
            description=(
                f"Sensitive path {path} returned HTTP "
                f"{ep.status_code or 'unknown'} — may expose internal "
                f"configuration, debug info, or admin interfaces."
            ),
        )]

    @staticmethod
    def _check_auth_boundary(ep: Endpoint) -> list[VulnIndicator]:
        if ep.requires_auth is not False:
            return []
        path = urlparse(ep.url).path
        path_lower = path.lower()
        data_patterns = (
            "/user", "/account", "/profile", "/transaction",
            "/balance", "/card", "/order", "/payment",
            "/message", "/notification",
        )
        if not any(seg in path_lower for seg in data_patterns):
            return []

        return [VulnIndicator(
            pattern=VulnPattern.AUTH_BOUNDARY_GAP,
            confidence=RiskLevel.HIGH,
            evidence=f"No auth required: {ep.method} {path}",
            description=(
                f"{ep.method} {path} returns data without authentication. "
                f"If this endpoint exposes user-specific resources, any "
                f"unauthenticated caller can access them."
            ),
        )]

    @staticmethod
    def _check_excessive_data(ep: Endpoint) -> list[VulnIndicator]:
        sensitive = {f.lower() for f in ep.response_fields} & _SENSITIVE_RESPONSE_FIELDS
        if not sensitive:
            return []
        path = urlparse(ep.url).path
        field_list = ", ".join(sorted(sensitive))
        return [VulnIndicator(
            pattern=VulnPattern.EXCESSIVE_DATA_EXPOSURE,
            confidence=RiskLevel.HIGH,
            evidence=f"Sensitive response fields: {field_list}",
            description=(
                f"Response schema for {ep.method} {path} includes "
                f"sensitive fields ({field_list}). Strip credentials "
                f"and PII before returning data to clients."
            ),
        )]

    @staticmethod
    def _check_response_body_leaks(ep: Endpoint) -> list[VulnIndicator]:
        """Detect sensitive data leaked in response body snippets."""
        body = ep.response_body_snippet
        if not body:
            return []

        path = urlparse(ep.url).path
        indicators: list[VulnIndicator] = []

        match = _STACK_TRACE_PATTERN.search(body)
        if match:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.HIGH,
                evidence=f"Stack trace in response: {match.group(0)[:80]}",
                description=(
                    f"Stack trace in {ep.method} {path} response leaks "
                    f"internal file paths and framework details."
                ),
            ))

        ips = _INTERNAL_IP_PATTERN.findall(body)
        if ips:
            ip_list = ", ".join(ips[:3])
            indicators.append(VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.MEDIUM,
                evidence=f"Internal IPs in response: {ip_list}",
                description=(
                    f"RFC 1918 addresses ({ip_list}) in the response from "
                    f"{ep.method} {path} reveal internal network layout."
                ),
            ))

        match = _SQL_FRAGMENT_PATTERN.search(body)
        if match:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.INFO_DISCLOSURE,
                confidence=RiskLevel.HIGH,
                evidence=f"SQL error in response: {match.group(0)[:80]}",
                description=(
                    f"SQL error in {ep.method} {path} output — suggests "
                    f"an injection surface and reveals the DB engine."
                ),
            ))

        match = _API_KEY_TOKEN_PATTERN.search(body)
        if match:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.EXCESSIVE_DATA_EXPOSURE,
                confidence=RiskLevel.CRITICAL,
                evidence=f"API key/token pattern: {match.group(0)[:30]}...",
                description=(
                    f"API key or token found in {ep.method} {path} response "
                    f"body — could be used to impersonate the service."
                ),
            ))

        emails = _EMAIL_PATTERN.findall(body)
        if len(emails) > _EXCESSIVE_EMAIL_THRESHOLD:
            indicators.append(VulnIndicator(
                pattern=VulnPattern.EXCESSIVE_DATA_EXPOSURE,
                confidence=RiskLevel.MEDIUM,
                evidence=f"Multiple emails in response: {', '.join(emails[:3])} + {len(emails)-3} more",
                description=(
                    f"{len(emails)} email addresses in {ep.method} {path} "
                    f"response — looks like over-fetching user records."
                ),
            ))

        return indicators

    @staticmethod
    def _check_injection_surfaces(ep: Endpoint) -> list[VulnIndicator]:
        """Detect params that suggest SQL injection, path traversal, or command injection."""
        indicators: list[VulnIndicator] = []
        path = urlparse(ep.url).path
        params = _collect_param_names(ep)

        sql_params = params & _SQL_INJECTION_PARAMS
        if sql_params:
            param_list = ", ".join(sorted(sql_params))
            indicators.append(VulnIndicator(
                pattern=VulnPattern.SQL_INJECTION,
                confidence=RiskLevel.MEDIUM,
                evidence=f"SQL-related params: {param_list}",
                description=(
                    f"Parameters ({param_list}) on {ep.method} {path} "
                    f"suggest dynamic query construction. Test for SQL "
                    f"injection via malformed input."
                ),
            ))

        traversal_params = params & _PATH_TRAVERSAL_PARAMS
        if traversal_params:
            param_list = ", ".join(sorted(traversal_params))
            indicators.append(VulnIndicator(
                pattern=VulnPattern.PATH_TRAVERSAL,
                confidence=RiskLevel.MEDIUM,
                evidence=f"File-path params: {param_list}",
                description=(
                    f"Parameters ({param_list}) on {ep.method} {path} "
                    f"reference files or paths. Test for directory traversal "
                    f"via ../ sequences."
                ),
            ))

        cmd_params = params & _COMMAND_INJECTION_PARAMS
        if cmd_params:
            param_list = ", ".join(sorted(cmd_params))
            indicators.append(VulnIndicator(
                pattern=VulnPattern.COMMAND_INJECTION,
                confidence=RiskLevel.HIGH,
                evidence=f"Command-related params: {param_list}",
                description=(
                    f"Parameters ({param_list}) on {ep.method} {path} "
                    f"suggest OS command execution. Test for command "
                    f"injection via shell metacharacters."
                ),
            ))

        return indicators

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
        insights_context = self.build_insights_context(state)
        strategy_context = ""
        if state.scan_strategy and state.scan_strategy.priority_patterns:
            strategy_context = (
                f"\nPriority Patterns (from planner): "
                f"{', '.join(state.scan_strategy.priority_patterns)}\n"
            )

        total_additions = 0
        for i in range(0, len(all_endpoints), LLM_VULN_BATCH_SIZE):
            batch = all_endpoints[i : i + LLM_VULN_BATCH_SIZE]
            try:
                batch_context = self._format_batch_for_llm(state, batch)
                messages = [
                    {"role": "system", "content": self.prompt_registry.get("vuln_analyzer").system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Target: {state.target}\n\n"
                            f"Prior Insights:\n{insights_context}\n\n"
                            f"Tech Stack:\n{tech_context}{strategy_context}\n\n"
                            f"Prior Findings:\n{findings_context}\n\n"
                            f"All Endpoints Summary (for cross-referencing):\n{cross_ref}\n\n"
                            f"BATCH TO ANALYZE ({len(batch)} endpoints):\n{batch_context}"
                        ),
                    },
                ]

                response = await self.llm.chat_structured(
                    messages, response_model=VulnAnalysisBatchResponse,
                    name=f"vuln_llm_batch_{i // LLM_VULN_BATCH_SIZE}",
                    confidence_threshold=settings.active_confidence_threshold,
                )
                total_additions += self._process_llm_vuln_results(state, response, batch)

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

            # include response body snippet for LLM context
            if ep.response_body_snippet:
                snippet = ep.response_body_snippet[:500]
                lines.append(f"  Response body (first 500 chars): {snippet}")

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
        state: ScanState, data: VulnAnalysisBatchResponse, batch: list[tuple[str, Endpoint]]
    ) -> int:
        """Merge LLM output back into state vuln_indicators."""
        total = 0
        batch_keys = {key for key, _ in batch}

        for analysis in data.endpoint_analyses:
            if analysis.endpoint_key not in batch_keys:
                continue

            existing = state.vuln_indicators.get(analysis.endpoint_key, [])
            _apply_suppressions(
                indicators=existing,
                suppressions=[s.model_dump() for s in analysis.suppressions],
            )
            _apply_confidence_adjustments(
                indicators=existing,
                adjustments=[a.model_dump() for a in analysis.confidence_adjustments],
            )
            total += _apply_new_indicators(
                state=state,
                existing=existing,
                ep_key=analysis.endpoint_key,
                new_indicators=[i.model_dump() for i in analysis.new_indicators],
            )
            state.vuln_indicators[analysis.endpoint_key] = existing

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
                    path = urlparse(ep.url).path
                    results.setdefault(key, []).append(VulnIndicator(
                        pattern=VulnPattern.API_VERSION_CONFUSION,
                        confidence=RiskLevel.HIGH,
                        evidence=(
                            f"Resource {resource} — v{ver} auth={ep.requires_auth}, "
                            f"other versions differ"
                        ),
                        description=(
                            f"{ep.method} {path} (v{ver}, auth={ep.requires_auth}) "
                            f"has inconsistent auth requirements compared to other "
                            f"API versions of {resource}. Older versions may lack "
                            f"security controls added in newer versions."
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
            status = f" [{ep.status_code}]" if ep.status_code else ""
            results.setdefault(key, []).append(VulnIndicator(
                pattern=VulnPattern.BROKEN_FUNCTION_LEVEL_AUTH,
                confidence=RiskLevel.CRITICAL,
                evidence=f"Admin/debug path without auth: {ep.method} {path}{status}",
                description=(
                    f"Admin/debug path {path} is accessible without auth "
                    f"(HTTP {ep.status_code or 'unknown'}). Could allow "
                    f"privilege escalation or access to operational data."
                ),
            ))
        return results
