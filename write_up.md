# Ghost API Hunter — Write-Up

## Discovery Strategy

I designed this as a multi-agent pipeline where deterministic discovery runs first, and then LLM-augmented passes build on top of that. The idea was to squeeze as much signal as possible out of traditional methods before bringing in the model.

So, passive recon agent comes first: robots.txt, sitemap.xml, headers, security.txt, and other well-known paths. After that, a crawler runs BFS from the base URL and collects links, forms, and script sources. In parallel, there’s API discovery and a JS analyzer. These handle things like OpenAPI probing, checking common API paths, version enumeration, and regex extraction of routes from JavaScript bundles.

On top of that, there’s a hypothesis agent. It takes the full scan context and generates educated guesses for endpoints that might exist but weren’t directly found. Those guesses are then validated with real HTTP requests.

I intentionally avoided brute-force wordlist fuzzing. Instead, I focused on fingerprinting, response analysis, crawling, header inspection, and parsing OpenAPI specs. The HTTP client includes adaptive rate limiting and backoff so it behaves reasonably against real-world targets.

For the model, I used Groq with Llama 3.3 70B. Speed was important because the pipeline makes a lot of LLM calls per run (hypothesis generation, classification, vuln analysis, prioritization, report synthesis, etc.). Groq’s inference speed keeps scans from feeling slow. I also needed reliable structured outputs (JSON schemas, Pydantic models), and the 70B model handled that more consistently than smaller ones. I tested 8B variants, but they had more parsing errors and weaker reasoning around endpoint guessing. I run everything at temperature 0.0 for deterministic, reproducible scans.

## LLM Integration

The LLM is involved in almost every major step: planner, API endpoint guessing, hypothesis generation, classifier, second-pass vuln analyzer, verifier, prioritizer, and report synthesis. Even the “wave insights” are LLM-generated.

Where it helped most was endpoint discovery. The API discovery and hypothesis agents found nested resource paths and admin-style variants that deterministic logic missed. The second pass of the vuln analyzer was useful for generating semantic variants, identifying chained issues, and suppressing obvious false positives.

The prioritizer adds rationale and even suggests curl-style tests, which makes the output much more actionable. The planner also adjusts focus areas and proposes extra paths based on early findings. Finally, report synthesis turns what would otherwise be raw JSON into something readable.

That said, it’s not perfect. The classifier and prioritizer sometimes return slightly wrong formats (for example, incorrect parameter names in generated test commands). The prioritizer can also run into token limits on large scans. Adding stricter validation and batching would help. Still, overall the LLM clearly extends discovery depth and improves usability compared to a purely rule-based approach.

## What Didn’t Work

The original version of this was mostly deterministic. I’ve been gradually expanding it toward a more AI-native design. It’s not fully there yet. To really qualify as AI-native, it would need persistent memory, agent-driven orchestration, and more model autonomy. Right now, it’s still a structured pipeline with LLM components layered in. But the current architecture sets up that transition.

## Self-Improvement Idea

One improvement would be feedback-driven prompt refinement. The system could learn from real outcomes across runs and targets without fine-tuning the model itself.

For example:
	1.	Log which LLM-generated endpoint guesses return 2xx vs 4xx/5xx responses. Aggregate successful patterns and turn them into few-shot examples.
	2.	When endpoints are validated later in the pipeline, track whether the classifier’s decision was correct. Use misclassifications to adjust prompts.
	3.	Track which common or “extra” paths produce findings for different domain types, and build a lightweight lookup table of effective paths per domain.
	4.	Maintain a small corpus of (domain_type, path, outcome) tuples and sample from it when generating new hypotheses.

Implementation-wise, each run would log outcomes in JSON. A separate “tune” step would update few-shot banks and domain-path tables. Future runs would load that context dynamically. No fine-tuning required — the system just gradually improves as it scans more targets.

## Toward AI-Native

To move meaningfully closer to AI-native behavior, I’d make three structural changes:
	•	Agentic orchestration: Replace the fixed DAG pipeline with a ReAct-style loop where the model decides the next action. Instead of “run these 10 agents in order,” the model would reason about the current state, choose a tool (crawl, probe, analyze JS, hypothesize, classify, verify), observe the result, and repeat.
	•	Persistent memory: Add a vector store or lightweight database of past scan outcomes (endpoints found, vulnerabilities, domain types, effective paths). The model could query things like, “For Flask-based banking apps, which paths tend to expose sensitive data?” and use that to guide discovery.
	•	Tool-augmented single agent: Instead of many semi-independent agents, expose tools (HTTP client, crawler, OpenAPI fetcher, JS analyzer) to a central reasoning agent. The model drives the workflow, deciding what to try next based on observations.