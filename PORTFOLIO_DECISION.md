# Portfolio decision: Switchback

Prepared September 11, 2026. Scores are engineering judgments about project potential, not measured outcomes or predictions of interview success. Assumptions: one DGX Spark, laptop development, public downloads, small budget, 12–16 focused build sessions plus unattended runs. Use May 2028 as the graduation date supplied in the current request.

## Step 1: five candidates

The largest gap is demonstrated ML execution and performance reasoning. Threadline already provides substantial systems-correctness evidence. Another service with queues, dashboards, and a large test count would add less. Kernel work is a separate remaining gap; Switchback's MVP does not pretend to fill it.

Criteria: Signal = 30-second README signal; Gap = portfolio complement; Metrics = reproducible measurement opportunity; Defensible = explainable implementation; Buildable = 2–4 week feasibility; Relevant = relevance to the stated target companies and roles. Equal weights, 10 is best.

| Candidate | Signal | Gap | Metrics | Defensible | Buildable | Relevant | Total / 60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Switchback: adaptive speculative decoding** | 9 | 10 | 9 | 8 | 8 | 10 | **54** |
| KernelScope: fused transformer operator lab | 9 | 10 | 10 | 7 | 6 | 10 | 52 |
| EvalLens: inference regression minimizer | 8 | 9 | 8 | 9 | 8 | 9 | 51 |
| CheckpointLab: memory-budgeted training planner | 8 | 9 | 9 | 8 | 7 | 9 | 50 |
| CacheCraft: bounded KV-cache inference runtime | 9 | 9 | 9 | 7 | 6 | 10 | 50 |

**1. Switchback.** Implement a small-model draft / large-model verification decoder, then use measured costs to choose lookahead or stop drafting. A token-level trace explains the result, and an independent finite-state oracle checks sampling behavior. **Hard core:** rejection sampling, causal-logit alignment, transactional KV rollback, and adaptive decisions that preserve sampling semantics. **Headline metric:** held-out warm-request latency speedup versus HF target-only, alongside HF dynamic speculation and calibration-selected fixed lookahead, with confidence intervals and slowdown rates. **Roles:** ML systems, research engineering, AI infrastructure SWE; also a useful technical deployment story for FDE. Dynamic speculation already exists, so the differentiator must be the correctness and measurement work. [Existing Hugging Face approach](https://huggingface.co/blog/dynamic_speculation_lookahead)

**2. KernelScope.** Build and profile a fused residual-add plus RMSNorm operator with a shape-aware dispatcher, then integrate it into one transformer inference path. The useful angle is proving where a custom operator beats both eager and compiled PyTorch and where it should fall back. **Hard core:** reductions, memory traffic, launch overhead, alignment/tail handling, numerical tolerances, and end-to-end integration. **Headline metric:** operator latency and full-model latency against eager and `torch.compile` baselines across a locked shape grid. **Roles:** NVIDIA GPU software, ML compilers, inference performance, research engineering. It adds the most low-level depth but requires new Triton or CUDA knowledge and more platform debugging than the schedule comfortably allows.

**3. EvalLens.** Build a differential inference tester that finds and minimizes regressions between eager, compiled, cached, and quantized execution of the same small transformer. The output is a minimal reproducible case with a tolerance-aware diagnosis, not an LLM judge score. **Hard core:** metamorphic properties, stateful cache failures, numerical error classification, delta debugging, and false-positive control. **Headline metric:** injected-fault detection recall at a fixed false-positive rate and reduction in failing-case size/time versus random testing and naive reduction. **Roles:** evaluation infrastructure, ML framework quality, devtools, research engineering. Its weakness is that seeded faults measure the harness, not proof of undiscovered real bugs.

**4. CheckpointLab.** Build a training-memory planner that profiles a small transformer, chooses activation checkpoint boundaries under a memory budget, and verifies gradients against an uncheckpointed reference. Optimize for a measured time/memory Pareto frontier. **Hard core:** activation liveness, recomputation costs, constrained planning, RNG preservation, and gradient equivalence. **Headline metric:** training-step latency at matched peak memory versus no checkpointing and uniform checkpointing, using a fixed public text workload. **Roles:** training infrastructure, ML systems, research engineering. It complements the portfolio well, but its reviewer demo is less immediately legible than token verification and latency traces.

**5. CacheCraft.** Build a bounded contiguous KV-cache runtime for batch-one inference, with exact full-cache and sliding-window modes, then quantify the latency/memory/quality tradeoff under long contexts. A replayable cache inspector explains eviction and recomputation decisions. **Hard core:** cache ownership, position encodings, eviction semantics, memory accounting, and matching baselines without claiming approximation is exact. **Headline metric:** peak memory and per-token latency at a specified next-token divergence or task-score budget versus a full-cache baseline. **Roles:** inference infrastructure and systems SWE. It risks expanding into a serving-engine clone, and cache approximation introduces an additional research problem in the same short schedule.

## Step 2: winner

Choose **Switchback** because it adds ML execution depth to the reliability skills already demonstrated by Threadline and gives a reviewer a concrete latency/correctness question. It uses existing Python/PyTorch strengths while requiring new understanding of sampling, caches, and GPU measurement, with no model training or paid infrastructure. It beats runner-up KernelScope on schedule and interview ownership: starting with custom GPU reductions and architecture tuning adds several new failure surfaces before the result becomes useful. The selection is conditional on evidence; if Switchback cannot demonstrate a compelling result, publish the honest study and retain Threadline as the resume lead.

## Step 3: build spec

See `SPEC.md`. It contains the architecture, exact sampling/cache contracts, model and workload choices, eight session-sized milestones, test invariants, benchmark matrix, metric formulas, demo, risks, and fallbacks. The supplied `scripts/render_results.py` renders actual future benchmark logs; it does not benchmark models or create plausible measurements.

## Step 4: Claude instructions

See `CLAUDE.md`. Copy the entire bundle into a new repo and start Claude Code with:

> Read CLAUDE.md, SPEC.md, and PROGRESS.md. Implement milestone 1 and continue through the milestones in order, making small local commits and updating PROGRESS.md after every milestone. Treat every GPU gate and benchmark as incomplete until it actually runs. Keep the measured scope honest and follow the documented fallbacks when needed.

The bundle contains planning documents and the report renderer. It does not contain the inference engine. Do not mistake this design delivery for a completed portfolio project.

## Step 5: resume templates

Fill these only from generated results and completed implementation. Bracketed items are slots, not claims. Each template is one line and uses no em dash.

- Built Switchback, a PyTorch speculative decoding engine with [METRIC]× warm-request speedup over Hugging Face target-only inference on [METRIC] held-out prompts using NVIDIA DGX Spark.
- Implemented rejection sampling and transactional KV-cache rollback, achieving [METRIC]% greedy token agreement and passing [METRIC] independently enumerated distribution checks.
- Developed a cost-based speculation controller that reduced median latency by [METRIC]% versus calibration-selected fixed lookahead while limiting >5% slowdowns to [METRIC]% of held-out prompts.

The third bullet is usable only if that comparison wins and the slowdown fraction is measured. Otherwise describe the break-even evaluation instead of claiming an improvement. Never write "zero quality loss" based only on greedy agreement or finite-model checks.

## Five hardest interview questions

| Question | What Ethan must understand |
|---|---|
| **1. Why does accepting a draft token with p/q, followed by a positive residual, reproduce the target distribution?** | Derive accepted mass `min(p(x), q(x))`; derive rejected mass times the normalized positive residual; show their sum is `p(x)`. Explain support, conditioning on an accepted prefix, the bonus token, and why greedy decoding is separate. Distinguish mathematical exactness from floating-point behavior and empirical tests. |
| **2. Walk through a rejection at the third candidate and show exactly what is in each KV cache.** | Use the pending-token convention, map all verification rows to prefixes, retain only two accepted candidates, commit the correction, and crop both caches to the new sequence excluding its final token. Explain the extra draft catch-up call when every proposal is accepted. Know why stale tentative keys can yield plausible but wrong outputs. |
| **3. Why can high acceptance still produce a slowdown, and how does your controller know when to bypass?** | Account for draft prefill, autoregressive draft cost, target verification width, residual calculation, launch/synchronization overhead, extra draft catch-up, and EOS. Derive expected emitted tokens from conditional acceptance. Explain censored observations, imperfect cost prediction, and the limits of sticky bypass. |
| **4. What would make your reported speedup misleading?** | Explain asynchronous GPU timing, warmup, output-length confounding, hidden model defaults, weaker baselines, thermal/order effects, compilation asymmetry, data leakage, and correlated repeats. Derive the geometric mean of per-prompt paired ratios and explain why the bootstrap resamples prompts. Name what the experiment does not measure. |
| **5. Why not just use HF assisted generation or vLLM, and what changes under load?** | Credit existing speculative decoding. Identify the actual contribution: independent correctness engine, inspectable rollback, measured controller, and reproducible evidence. Explain that batching can change target efficiency and draft contention, why batch-one gains need not survive serving load, and what production features are absent. Be able to explain major code paths even though Claude Code implemented them. |

## Practical ownership requirement

Reserve the last 10–15 minutes of each build session for a teach-back: draw the state, derive one decision, and reproduce one failing test. Those are practice estimates, not a coding milestone. Do not lead the resume with this project until Ethan can answer the sampling proof, cache rejection, and benchmark-validity questions without relying on generated prose.
