"""API discovery — OpenAPI spec parsing, common paths, version enumeration, and LLM guessing."""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import urljoin

from src.ghost_hunter.agents.registry import register_agent
from src.ghost_hunter.agents.base import BaseAgent
from src.ghost_hunter.models import (
    AgentResult,
    DiscoverySource,
    Endpoint,
    Finding,
    ParamLocation,
    ParameterDetail,
    RiskLevel,
    ScanState,
    SecuritySchemeInfo,
)

logger = logging.getLogger(__name__)

MAX_API_VERSION = 3
MAX_URLS_FOR_LLM_CONTEXT = 50
MAX_LLM_GUESSES_TO_VALIDATE = 20
MAX_CORS_CHECKS = 30

PERMISSIVE_CORS_ORIGINS = {"*", "null"}

OPENAPI_PATHS = [
    "/openapi.json",
    "/swagger.json",
    "/api/docs",
    "/api/v1/docs",
    "/api/v2/docs",
    "/docs",
    "/swagger-ui.html",
    "/api-docs",
    "/api/swagger.json",
    "/api/openapi.json",
    "/redoc",
]

COMMON_API_PATHS = [
    "/api",
    "/api/v1",
    "/api/v2",
    "/api/users",
    "/api/user",
    "/api/accounts",
    "/api/account",
    "/api/admin",
    "/api/auth",
    "/api/login",
    "/api/register",
    "/api/logout",
    "/api/health",
    "/api/status",
    "/api/config",
    "/api/settings",
    "/api/search",
    "/api/upload",
    "/api/files",
    "/api/messages",
    "/api/notifications",
    "/api/payments",
    "/api/transactions",
    "/api/orders",
    "/api/products",
    "/api/profile",
    "/graphql",
    "/graphiql",
    "/admin",
    "/admin/login",
    "/debug",
    "/metrics",
    "/healthz",
    "/readyz",
    "/actuator",
    "/actuator/health",
    "/console",
    "/.env",
    "/wp-admin",
    "/wp-login.php",
    "/phpmyadmin",
]


_PARAM_LOCATION_MAP = {
    "path": ParamLocation.PATH,
    "query": ParamLocation.QUERY,
    "header": ParamLocation.HEADER,
    "cookie": ParamLocation.COOKIE,
    "body": ParamLocation.BODY,
    "formData": ParamLocation.BODY,
}


def _extract_field_names(
    schema: dict | None, definitions: dict | None = None, max_depth: int = 3
) -> list[str]:
    """Recursively pull property names from an OpenAPI schema, resolving $ref."""
    if not schema or max_depth <= 0:
        return []

    # resolve $ref
    ref = schema.get("$ref", "")
    if ref and definitions:
        ref_name = ref.rsplit("/", 1)[-1]
        schema = definitions.get(ref_name, {})
        if not schema:
            return []

    fields: list[str] = []
    props = schema.get("properties", {})
    for name, prop in props.items():
        fields.append(name)
        if prop.get("type") == "object":
            fields.extend(_extract_field_names(prop, definitions, max_depth - 1))

    # allOf / oneOf / anyOf
    for compose_key in ("allOf", "oneOf", "anyOf"):
        for sub in schema.get(compose_key, []):
            fields.extend(_extract_field_names(sub, definitions, max_depth - 1))

    # array items
    items = schema.get("items")
    if isinstance(items, dict):
        fields.extend(_extract_field_names(items, definitions, max_depth - 1))

    return list(dict.fromkeys(fields))  # dedupe preserving order


@register_agent
class APIDiscoveryAgent(BaseAgent):
    name = "api_discovery"
    description = (
        "Probes for OpenAPI specs, common API paths, performs version enumeration, "
        "and uses LLM to guess additional endpoints based on discovered patterns."
    )

    async def run(self, state: ScanState) -> AgentResult:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        spec_eps, spec_findings = await self._probe_openapi(state)
        endpoints.extend(spec_eps)
        findings.extend(spec_findings)

        common_eps = await self._probe_common_paths(state)
        endpoints.extend(common_eps)

        version_eps = await self._version_enumerate(state, endpoints)
        endpoints.extend(version_eps)

        llm_eps, llm_findings, llm_errors = await self._llm_guess_endpoints(state, endpoints)
        endpoints.extend(llm_eps)
        findings.extend(llm_findings)
        errors.extend(llm_errors)

        cors_findings = await self._check_cors_and_methods(state, endpoints)
        findings.extend(cors_findings)

        return AgentResult(
            agent_name=self.name,
            success=True,
            endpoints_found=endpoints,
            findings=findings,
            errors=errors,
        )

    async def _probe_openapi(
        self, state: ScanState
    ) -> tuple[list[Endpoint], list[Finding]]:
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []

        for path in OPENAPI_PATHS:
            resp = await self.http.get(path)
            if resp is None or resp.status_code not in (200, 301, 302):
                continue

            endpoints.append(
                Endpoint(
                    url=self.http.resolve_url(path),
                    method="GET",
                    status_code=resp.status_code,
                    content_type=resp.headers.get("content-type", ""),
                    discovered_by=DiscoverySource.OPENAPI_SPEC,
                    notes="OpenAPI/Swagger spec endpoint",
                )
            )

            if resp.status_code != 200:
                continue

            content_type = resp.headers.get("content-type", "")
            spec = None

            if "json" in content_type:
                try:
                    spec = resp.json()
                except json.JSONDecodeError as e:
                    logger.debug("Failed to parse JSON spec at %s: %s", path, e)
            elif "html" in content_type:
                spec = await self._extract_spec_from_swagger_ui(resp.text, path)

            if spec is None:
                continue

            spec_eps = self._parse_openapi_spec(spec)
            endpoints.extend(spec_eps)
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="openapi_spec_found",
                    title=f"OpenAPI spec found at {path}",
                    detail=f"Extracted {len(spec_eps)} endpoints from spec",
                    severity=RiskLevel.MEDIUM,
                    evidence=f"Title: {spec.get('info', {}).get('title', 'N/A')}",
                )
            )

        return endpoints, findings

    # patterns that Swagger UI uses to reference the spec URL (handles both quoted and unquoted keys)
    _SWAGGER_URL_PATTERNS = [
        re.compile(r""""?url"?\s*[:=]\s*['"]([^'"]+\.(?:json|yaml|yml))['"]"""),
        re.compile(r""""?spec[Uu]rl"?\s*[:=]\s*['"]([^'"]+)['"]"""),
        re.compile(r""""?swagger[Uu]rl"?\s*[:=]\s*['"]([^'"]+)['"]"""),
        re.compile(r""""?configUrl"?\s*[:=]\s*['"]([^'"]+)['"]"""),
    ]

    async def _extract_spec_from_swagger_ui(
        self, html: str, page_path: str
    ) -> dict | None:
        """Extract and fetch the OpenAPI spec JSON from a Swagger UI HTML page."""
        for pattern in self._SWAGGER_URL_PATTERNS:
            match = pattern.search(html)
            if not match:
                continue
            spec_url = match.group(1)
            if not spec_url.startswith(("http://", "https://")):
                spec_url = urljoin(self.http.resolve_url(page_path), spec_url)
            resp = await self.http.get(spec_url)
            if resp and resp.status_code == 200:
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    logger.debug("Swagger UI spec URL %s returned non-JSON", spec_url)
        return None

    _HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}

    def _parse_openapi_spec(self, spec: dict) -> list[Endpoint]:
        """Extract endpoints from an OpenAPI 2.x or 3.x spec."""
        endpoints: list[Endpoint] = []
        paths = spec.get("paths", {})
        base_path = spec.get("basePath", "")

        for path, methods in paths.items():
            full_path = base_path + path if base_path else path
            for method in methods:
                if method.lower() not in self._HTTP_METHODS:
                    continue
                endpoints.append(
                    self._parse_operation(full_path, method, methods[method], spec)
                )

        return endpoints

    def _parse_operation(
        self, path: str, method: str, operation: dict | object, spec: dict | None = None
    ) -> Endpoint:
        if not isinstance(operation, dict):
            operation = {}
        spec = spec or {}

        # definitions for $ref resolution (Swagger 2.x vs OpenAPI 3.x)
        definitions = spec.get("definitions") or spec.get("components", {}).get("schemas", {})

        # basic param names (backward compat)
        params = [
            p["name"] for p in operation.get("parameters", [])
            if isinstance(p, dict) and p.get("name")
        ]

        # rich parameter details
        param_details: list[ParameterDetail] = []
        for p in operation.get("parameters", []):
            if not isinstance(p, dict) or not p.get("name"):
                continue
            loc_str = p.get("in", "query")
            loc = _PARAM_LOCATION_MAP.get(loc_str, ParamLocation.QUERY)
            param_details.append(ParameterDetail(
                name=p["name"],
                location=loc,
                param_type=p.get("type", p.get("schema", {}).get("type", "string")),
                required=p.get("required", False),
            ))

        # request body (OpenAPI 3.x)
        body_content_type, body_fields = self._parse_request_body(operation, definitions)

        # Swagger 2.x body params
        if not body_fields:
            for p in operation.get("parameters", []):
                if isinstance(p, dict) and p.get("in") == "body":
                    schema = p.get("schema", {})
                    body_fields = _extract_field_names(schema, definitions)
                    break

        # response fields
        resp_fields = self._parse_response_fields(operation, definitions)

        # security schemes
        security_schemes = self._parse_security(operation, spec)

        return Endpoint(
            url=self.http.resolve_url(path),
            method=method.upper(),
            discovered_by=DiscoverySource.OPENAPI_SPEC,
            parameters=params,
            parameter_details=param_details,
            request_body_content_type=body_content_type,
            request_body_fields=body_fields,
            response_fields=resp_fields,
            security_schemes=security_schemes,
            notes=operation.get("summary", ""),
        )

    @staticmethod
    def _parse_request_body(
        operation: dict, definitions: dict | None
    ) -> tuple[str | None, list[str]]:
        """Extract content type and field names from OpenAPI 3.x requestBody."""
        req_body = operation.get("requestBody", {})
        if not req_body:
            return None, []
        content = req_body.get("content", {})
        for ct, media in content.items():
            schema = media.get("schema", {})
            fields = _extract_field_names(schema, definitions)
            return ct, fields
        return None, []

    @staticmethod
    def _parse_response_fields(operation: dict, definitions: dict | None) -> list[str]:
        """Extract field names from the success response schema."""
        responses = operation.get("responses", {})
        for code in ("200", "201", "default"):
            resp = responses.get(code, {})
            if not resp:
                continue
            # OpenAPI 3.x
            content = resp.get("content", {})
            for media in content.values():
                schema = media.get("schema", {})
                fields = _extract_field_names(schema, definitions)
                if fields:
                    return fields
            # Swagger 2.x
            schema = resp.get("schema", {})
            if schema:
                fields = _extract_field_names(schema, definitions)
                if fields:
                    return fields
        return []

    @staticmethod
    def _parse_security(operation: dict, spec: dict) -> list[SecuritySchemeInfo]:
        """Extract security scheme info from operation or spec level."""
        schemes: list[SecuritySchemeInfo] = []
        # security requirement at operation or spec level
        security = operation.get("security") or spec.get("security", [])
        if not security:
            return schemes

        # scheme definitions (OpenAPI 3.x vs Swagger 2.x)
        scheme_defs = (
            spec.get("components", {}).get("securitySchemes", {})
            or spec.get("securityDefinitions", {})
        )
        if not scheme_defs:
            return schemes

        seen: set[str] = set()
        for req in security:
            if not isinstance(req, dict):
                continue
            for scheme_name in req:
                if scheme_name in seen:
                    continue
                seen.add(scheme_name)
                defn = scheme_defs.get(scheme_name, {})
                if not isinstance(defn, dict):
                    continue
                schemes.append(SecuritySchemeInfo(
                    scheme_type=defn.get("type", "unknown"),
                    scheme_name=scheme_name,
                    location=defn.get("in", ""),
                    bearer_format=defn.get("bearerFormat", ""),
                ))

        return schemes

    async def _probe_common_paths(self, state: ScanState) -> list[Endpoint]:
        endpoints: list[Endpoint] = []

        for path in COMMON_API_PATHS:
            key = state.endpoint_key("GET", self.http.resolve_url(path))
            if key in state.endpoints:
                continue

            resp = await self.http.get(path)
            if resp is None:
                continue

            if resp.status_code < 404:
                endpoints.append(
                    Endpoint(
                        url=self.http.resolve_url(path),
                        method="GET",
                        status_code=resp.status_code,
                        content_type=resp.headers.get("content-type", ""),
                        discovered_by=DiscoverySource.COMMON_PATH,
                    )
                )
                if resp.status_code == 403:
                    state.blocked_paths.append(path)

        return endpoints

    async def _version_enumerate(
        self, state: ScanState, new_endpoints: list[Endpoint]
    ) -> list[Endpoint]:
        """If /api/v1/X exists, try /api/v2/X and vice versa."""
        version_eps: list[Endpoint] = []
        version_pattern = re.compile(r"(/api/v)(\d+)(/.*)")

        all_urls = [ep.url for ep in new_endpoints] + [
            ep.url for ep in state.endpoints.values()
        ]

        tried: set[str] = set()

        for url in all_urls:
            path = url.replace(self.http.base_url, "")
            match = version_pattern.search(path)
            if not match:
                continue

            prefix, version, suffix = match.group(1), int(match.group(2)), match.group(3)

            for alt_version in range(1, MAX_API_VERSION + 1):
                if alt_version == version:
                    continue
                alt_path = f"{prefix}{alt_version}{suffix}"
                if alt_path in tried:
                    continue
                tried.add(alt_path)

                resp = await self.http.head(alt_path)
                if resp and resp.status_code < 404:
                    version_eps.append(
                        Endpoint(
                            url=self.http.resolve_url(alt_path),
                            method="GET",
                            status_code=resp.status_code,
                            discovered_by=DiscoverySource.VERSION_ENUM,
                            api_version=f"v{alt_version}",
                            notes=f"version enum from v{version}",
                        )
                    )

        return version_eps

    async def _llm_guess_endpoints(
        self, state: ScanState, new_endpoints: list[Endpoint]
    ) -> tuple[list[Endpoint], list[Finding], list[str]]:
        """Use LLM to guess additional API endpoints based on discovered patterns."""
        endpoints: list[Endpoint] = []
        findings: list[Finding] = []
        errors: list[str] = []

        known_urls = [ep.url for ep in new_endpoints] + [
            ep.url for ep in state.endpoints.values()
        ]
        if not known_urls:
            return endpoints, findings, errors

        known_urls = sorted(set(known_urls))[:MAX_URLS_FOR_LLM_CONTEXT]

        tech_info = self.build_tech_context(state)
        findings_context = self.build_findings_context(state)

        messages = [
            {
                "role": "system",
                "content": (
                    "CONTEXT:\n"
                    "You are part of the API discovery pipeline. OpenAPI spec parsing and common path "
                    "probing have already run. Your guesses should fill gaps those methods missed — "
                    "undocumented endpoints, internal APIs, deprecated routes, and resource sub-paths.\n\n"
                    "ROLE:\n"
                    "You are an API security researcher who reverse-engineers web applications. You "
                    "understand how developers structure APIs and where they leave undocumented endpoints.\n\n"
                    "ACTION:\n"
                    "Follow this 6-step reasoning process:\n"
                    "1. NAMING SCHEME: Identify the API naming convention (snake_case, camelCase, "
                    "kebab-case, plural vs singular) and match it in your guesses\n"
                    "2. COMPLETE CRUD: For each resource with GET, check for POST/PUT/PATCH/DELETE. "
                    "For collections, check for /{id} sub-resource\n"
                    "3. INFER SUB-RESOURCES: /users/{id}/orders, /accounts/{id}/transactions — "
                    "standard nested resource patterns\n"
                    "4. ADMIN/INTERNAL VARIANTS: /admin/users, /internal/metrics, /debug/logs — "
                    "endpoints developers forget to protect\n"
                    "5. TECH-STACK PATHS: Framework-specific endpoints (Django /admin/, Spring "
                    "/actuator/env, Express /debug, Flask /_debug_toolbar/)\n"
                    "6. DEPRECATED ENDPOINTS: Check for /v1/ equivalents of /v2/ endpoints, "
                    "/old/, /legacy/ prefixed paths\n\n"
                    "FORMAT:\n"
                    "Respond with JSON:\n"
                    "{\n"
                    '  "reasoning": "2-3 sentences about patterns and gaps observed",\n'
                    '  "guesses": [\n'
                    '    {"path": "/api/...", "method": "GET", "reason": "why this likely exists"}\n'
                    "  ]\n"
                    "}\n\n"
                    "FEW-SHOT EXAMPLES:\n"
                    "Given: GET /api/v1/users, POST /api/v1/users, GET /api/v1/orders\n"
                    "Good guesses:\n"
                    '  {"path": "/api/v1/users/{id}", "method": "GET", "reason": "Collection exists, '
                    'individual resource lookup expected"}\n'
                    '  {"path": "/api/v1/users/{id}", "method": "DELETE", "reason": "CRUD — create exists, '
                    'delete likely available"}\n'
                    '  {"path": "/api/v1/orders/{id}", "method": "GET", "reason": "Orders collection exists, '
                    'individual order retrieval expected"}\n\n'
                    "Keep it to 10-20 high-confidence guesses."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target: {state.target}\n\n"
                    f"Tech stack:\n{tech_info}\n\n"
                    f"Prior Findings:\n{findings_context}\n\n"
                    f"Discovered endpoints:\n" + "\n".join(f"  {u}" for u in known_urls)
                ),
            },
        ]

        try:
            data = await self.llm.chat_json(messages, name="api_discovery_llm_guess")
            guesses = data.get("guesses", [])

            validated = 0
            for guess in guesses[:MAX_LLM_GUESSES_TO_VALIDATE]:
                path = guess.get("path", "")
                method = guess.get("method", "GET").upper()
                if not path:
                    continue

                resp = await self.http.request(method if method == "GET" else "HEAD", path)
                if resp and resp.status_code < 404:
                    endpoints.append(
                        Endpoint(
                            url=self.http.resolve_url(path),
                            method=method,
                            status_code=resp.status_code,
                            content_type=resp.headers.get("content-type", ""),
                            discovered_by=DiscoverySource.LLM_API_GUESS,
                            notes=f"LLM guess: {guess.get('reason', '')}",
                        )
                    )
                    validated += 1

            if validated:
                findings.append(
                    Finding(
                        agent_name=self.name,
                        finding_type="llm_api_guesses_validated",
                        title=f"LLM guessed {validated} valid endpoints",
                        detail=f"Out of {len(guesses)} guesses, {validated} returned non-404 responses.",
                        severity=RiskLevel.INFO,
                    )
                )

        except Exception as e:
            errors.append(f"LLM endpoint guessing failed: {e}")

        return endpoints, findings, errors

    async def _check_cors_and_methods(
        self, state: ScanState, new_endpoints: list[Endpoint]
    ) -> list[Finding]:
        """Send OPTIONS to API endpoints; check CORS headers and allowed methods."""
        findings: list[Finding] = []
        permissive_cors: list[str] = []

        # deduplicate URLs across state + newly discovered
        urls_to_check = list({
            ep.url for ep in list(state.endpoints.values()) + new_endpoints
            if "/api/" in ep.url or "/graphql" in ep.url
        })[:MAX_CORS_CHECKS]

        last_origin = ""
        for url in urls_to_check:
            resp = await self.http.request("OPTIONS", url)
            if resp is None:
                continue

            allow_origin = resp.headers.get("access-control-allow-origin", "")
            allow_methods = resp.headers.get("access-control-allow-methods", "")

            if allow_origin in PERMISSIVE_CORS_ORIGINS:
                permissive_cors.append(url)
                last_origin = allow_origin

            if allow_methods:
                ep = state.find_endpoint("GET", url) or state.find_endpoint("POST", url)
                if ep:
                    ep.notes = (
                        f"{ep.notes}; allowed_methods={allow_methods}"
                        if ep.notes else f"allowed_methods={allow_methods}"
                    )

        if permissive_cors:
            findings.append(
                Finding(
                    agent_name=self.name,
                    finding_type="permissive_cors",
                    title=f"Permissive CORS on {len(permissive_cors)} endpoint(s)",
                    detail=(
                        "Access-Control-Allow-Origin: * allows any origin to make cross-site "
                        "requests. Endpoints: " + ", ".join(permissive_cors[:5])
                    ),
                    severity=RiskLevel.HIGH,
                    evidence=f"Access-Control-Allow-Origin: {last_origin}",
                )
            )

        return findings
