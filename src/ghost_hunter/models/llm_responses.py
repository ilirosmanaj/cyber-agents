"""Pydantic response models for all LLM call sites.

Each model validates the JSON returned by chat_structured(), replacing
fragile .get("key", default) access with typed fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 2a. Vuln Analyzer — VulnAnalysisBatchResponse
# ---------------------------------------------------------------------------

class VulnNewIndicator(BaseModel):
    pattern: str
    confidence: str = "medium"
    evidence: str = ""
    description: str = ""
    chain_with: str | None = None


class VulnSuppression(BaseModel):
    original_pattern: str
    reason: str = ""


class VulnConfidenceAdjustment(BaseModel):
    original_pattern: str
    new_confidence: str
    reason: str = ""


class VulnEndpointAnalysis(BaseModel):
    endpoint_key: str
    new_indicators: list[VulnNewIndicator] = Field(default_factory=list)
    suppressions: list[VulnSuppression] = Field(default_factory=list)
    confidence_adjustments: list[VulnConfidenceAdjustment] = Field(default_factory=list)


class VulnAnalysisBatchResponse(BaseModel):
    reasoning: str = ""
    endpoint_analyses: list[VulnEndpointAnalysis] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# 2b. Classifier — ClassificationBatchResponse
# ---------------------------------------------------------------------------

class EndpointClassification(BaseModel):
    index: int | None = None
    category: str = "unknown"
    requires_auth: bool | None = None
    url: str = ""
    method: str = "GET"


class ClassificationBatchResponse(BaseModel):
    reasoning: str = ""
    classifications: list[EndpointClassification] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# 2c. Hypothesis — HypothesisResponse
# ---------------------------------------------------------------------------

class EndpointHypothesis(BaseModel):
    path: str
    method: str = "GET"
    confidence: str = "medium"
    reasoning: str = ""


class HypothesisResponse(BaseModel):
    reasoning: str = ""
    hypotheses: list[EndpointHypothesis] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2d. Prioritizer — PrioritizationBatchResponse
# ---------------------------------------------------------------------------

class AttackSurfaceItem(BaseModel):
    index: int | None = None
    category: str = "unknown"
    risk_level: str = "info"
    rationale: str = ""
    suggested_tests: list[str] = Field(default_factory=list)
    url: str = ""
    method: str = "GET"


class PrioritizationBatchResponse(BaseModel):
    reasoning: str = ""
    attack_surface: list[AttackSurfaceItem] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# 2e. Verifier — VerifierBatchResponse
# ---------------------------------------------------------------------------

class VerifierAction(BaseModel):
    endpoint_key: str
    action: str  # "suppress" | "adjust_confidence" | "annotate"
    pattern: str
    new_confidence: str = ""
    reason: str = ""


class ReanalysisRequest(BaseModel):
    endpoint_key: str
    target_agent: str  # "classifier" or "vuln_analyzer"
    reason: str = ""


class VerifierBatchResponse(BaseModel):
    reasoning: str = ""
    actions: list[VerifierAction] = Field(default_factory=list)
    cross_cutting_notes: list[str] = Field(default_factory=list)
    reanalysis_requests: list[ReanalysisRequest] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# 2g. JS Analyzer — JSAnalysisBatchResponse
# ---------------------------------------------------------------------------

class JSEndpoint(BaseModel):
    path: str
    method: str = "GET"
    evidence: str = ""


class JSAnalysisBatchResponse(BaseModel):
    endpoints: list[JSEndpoint] = Field(default_factory=list)
    auth_patterns: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# 2h. API Discovery — APIGuessResponse
# ---------------------------------------------------------------------------

class APIGuess(BaseModel):
    path: str
    method: str = "GET"
    reason: str = ""


class APIGuessResponse(BaseModel):
    reasoning: str = ""
    guesses: list[APIGuess] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2i. Vuln Analyzer — ResponseBodyAnalysisResponse (Pass 1.5)
# ---------------------------------------------------------------------------

class ResponseBodyFinding(BaseModel):
    finding_type: str  # "secret", "config_leak", "debug_info", "credential"
    value_redacted: str  # the secret with middle chars masked
    context: str  # surrounding context explaining what this is
    confidence: str = "high"
    is_placeholder: bool = False  # true if it looks like a dummy/example value


class ResponseBodyAnalysisResponse(BaseModel):
    reasoning: str = ""
    findings: list[ResponseBodyFinding] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# 2j. Web Crawler — HTMLIntelAnalysisResponse
# ---------------------------------------------------------------------------

class HTMLIntelFinding(BaseModel):
    finding_type: str  # "leaked_secret", "sensitive_comment", "debug_indicator"
    evidence: str  # the relevant snippet
    context: str  # explanation of what was found and why it matters
    confidence: str = "high"
    is_placeholder: bool = False  # true if it looks like a dummy/example value


class HTMLIntelAnalysisResponse(BaseModel):
    reasoning: str = ""
    findings: list[HTMLIntelFinding] = Field(default_factory=list)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# 2k. Passive Recon — ReconAnalysisResponse
# ---------------------------------------------------------------------------

class ReconAnalysisResponse(BaseModel):
    reasoning: str = ""
    header_assessment: str = ""
    tech_hypotheses: list[str] = Field(default_factory=list)
    interesting_patterns: list[str] = Field(default_factory=list)
    initial_attack_vectors: list[str] = Field(default_factory=list)
    confidence: float | None = None
