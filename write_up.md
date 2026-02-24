# Ghost API Hunter — Write-Up

## Discovery Strategy

I designed this as a multi-agent pipeline where deterministic discovery runs first, and then LLM-augmented passes build on top of that. The ordering is deliberate: cheap and fast methods run first to build context, so that expensive LLM calls later have something meaningful to reason about.

Passive recon comes first: robots.txt, sitemap.xml, headers, security.txt, and other well-known paths. After that, a crawler runs BFS from the base URL and collects links, forms, and script sources. In parallel, there's API discovery and a JS analyzer. These handle things like OpenAPI probing, checking common API paths, version enumeration, and regex extraction of routes from JavaScript bundles.

On top of that, there's a hypothesis agent. It takes the full scan context and generates educated guesses for endpoints that might exist but weren't directly found. For example, after seeing `/api/v1/users`, it might guess `/api/v1/admin/users` — something pure crawling would never link to. Those guesses are then validated with real HTTP requests.

I intentionally avoided brute-force wordlist fuzzing. Instead, I focused on fingerprinting, response analysis, crawling, header inspection, and parsing OpenAPI specs. The HTTP client includes adaptive rate limiting and backoff so it behaves reasonably against real-world targets.

For the model, I used Groq with Llama 3.3 70B. Speed was important because the pipeline makes a lot of LLM calls per run (hypothesis generation, classification, vuln analysis, prioritization, report synthesis, etc.). Groq's inference speed keeps scans from feeling slow. I also needed reliable structured outputs (JSON schemas, Pydantic models), and the 70B model handled that more consistently than smaller ones. I tested 8B variants, but they had more parsing errors and weaker reasoning around endpoint guessing. I run everything at temperature 0.0 for deterministic, reproducible scans.

## LLM Integration

The LLM is involved in almost every major step: planner, API endpoint guessing, hypothesis generation, classifier, second-pass vuln analyzer, verifier, prioritizer, and report synthesis. Even the "wave insights" are LLM-generated.

Where it helped most was endpoint discovery. The API discovery and hypothesis agents found nested resource paths and admin-style variants that deterministic logic missed. The second pass of the vuln analyzer was useful for generating semantic variants, identifying chained issues, and suppressing obvious false positives.

The prioritizer adds rationale and even suggests curl-style tests, which makes the output much more actionable. The planner also adjusts focus areas and proposes extra paths based on early findings. Finally, report synthesis turns what would otherwise be raw JSON into something readable.

That said, it's not perfect. The classifier and prioritizer sometimes return slightly wrong formats (for example, incorrect parameter names in generated test commands). The prioritizer can also run into token limits on large scans. To mitigate format issues, all LLM calls use Pydantic models for structured output validation, which catches most malformed responses before they propagate. Batching large endpoint sets also helps stay within token limits. Overall the LLM clearly extends discovery depth and improves usability compared to a purely rule-based approach.

## What Didn't Work

Early on, I tried using hardcoded regex patterns for open-ended content analysis — things like detecting secrets in response bodies (`SECRET_KEY = 'value'` in a Werkzeug debug page) and flagging sensitive HTML comments or debug indicators in crawled pages. The regexes worked for the exact formats I wrote them for, but missed anything slightly different. A database URL with credentials, an API key in a non-standard format, a comment hinting at an auth bypass — all invisible to a fixed pattern. I replaced those with LLM passes that analyze the same content semantically, which generalized much better.

I also tested smaller models (8B parameter variants) early on. They were faster but produced more JSON parsing errors and weaker reasoning, especially for hypothesis generation where the model needs to infer plausible endpoints from context. The 70B model was the sweet spot between speed and reliability.

The architecture itself is still a structured pipeline with LLM components layered in, not a fully autonomous agent. To get there, it would need agentic orchestration (a ReAct-style loop where the model decides what to do next) and persistent memory across runs. The current design sets up that transition but isn't there yet.

## Self-Improvement Idea

One improvement would be feedback-driven prompt refinement. The system could learn from real outcomes across runs without fine-tuning the model itself.

For example:
	1.	Log which LLM-generated endpoint guesses return 2xx vs 4xx/5xx responses. Aggregate successful patterns into few-shot examples, and build a lightweight lookup of effective paths per domain type. Future hypothesis generation would sample from this corpus.
	2.	When endpoints are validated later in the pipeline, track whether the classifier's decision was correct. Use misclassifications to adjust prompts.
	3.	Feed this into a persistent memory layer — a vector store or simple database of past scan outcomes. The model could query it for things like "which paths tend to expose sensitive data on Flask apps?" and use that to prioritize discovery.

Implementation-wise, each run would log outcomes in JSON. A separate "tune" step would update few-shot banks and domain-path tables. Future runs would load that context dynamically. No fine-tuning required — the system just gradually improves as it scans more targets.
