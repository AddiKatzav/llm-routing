# A Simple If-Statement Beat Our AI System

*Part 3 of 3 — LLM Routing in Practice*

*Previously: [Part 1 — Your Local AI Has a Hidden Cliff](./article_1_the_cliff.md) · [Part 2 — We Built a Smart Fix. It Worked. It Was Also 70× Too Slow.](./article_2_too_slow.md)*

---

The most interesting result from this experiment wasn't the 36-point improvement in wall-avoidance score. It wasn't the 10-second latency disaster either.

It was what happened when we asked: **what if we never called the AI judge at all?**

---

## One Parameter, Dramatic Consequences

Our context-aware router had a sensitivity dial — a threshold that controlled when to invoke the AI decision-maker.

At threshold=0.85 (the default): the router called the AI judge on 73% of all routing decisions. Worst-case routing delay: 10.4 seconds.

We ran the same benchmark at four threshold settings and measured what changed.

![Worst-case routing delay (p95) at four threshold settings, on a log scale. Threshold=0.85: 10,437ms. Threshold=0.80: 6,211ms. Threshold=0.65: 17,914ms. Threshold=0.50: 0.04ms. A red line marks the 150ms spec. Only the 0.50 bar falls below it.](./plots/article_figB.png)

*Worst-case routing delay at four threshold settings (log scale — each tick is 10× the previous). Three out of four configurations miss the 150ms target by 40–120×. Then something unusual happens at threshold=0.50: the delay drops to 0.04 milliseconds — three million times faster than threshold=0.85. This is not a gradual improvement. It's a discontinuous jump.*

The drop from threshold=0.65 to threshold=0.50 is **440,000× faster**. That's not a smooth curve. It's a phase transition.

---

## Why the Jump Is Discontinuous

Here's what's happening.

Our benchmark tasks start at around 50% context occupancy. At threshold=0.65, a session starting at 50% is below the threshold — so it falls through to the AI judge. Same at 0.80 and 0.85. All three configurations still invoke the judge on the same 73% of turns.

But at threshold=0.50, something different happens: the rule fires first. A session starting at 50% occupancy hits the rule immediately — *escalate* — before the judge even gets a chance to run. The AI decision-maker is never called.

**At threshold=0.50: 0% of routing decisions invoke the AI judge.** Pure rule-based routing, every time. Less than 1 millisecond per decision.

---

## The Paradox

Here's the part that made us stop and think.

We set out to build a *context-aware, AI-powered router* — a system that uses machine learning to make smarter routing decisions than simple rules could. We ran a threshold sweep to find the optimal setting. And we found that the best configuration — the one that meets our latency spec — is the one where the AI component never runs.

The spec-compliant version of our "intelligent" router is this:

```python
if context_used >= 0.50:
    route_to_cloud()
else:
    route_to_local()
```

That's it. One comparison. No model inference. No neural network. No AI.

And here's the kicker: **the quality result is preserved.** The wall-avoidance score at threshold=0.50 is still 100% at near-wall depth — a 40-point improvement over static routing. Same outcome, achieved by a single conditional check instead of a language model.

---

## The Full Picture

![Pareto scatter plot showing p95 routing latency (x-axis, log scale) vs. wall-avoidance improvement (y-axis) for every router configuration we tested. A green zone marks the spec-compliant region (under 150ms latency, over +15pp improvement). Context-aware configurations at threshold=0.65, 0.80, and 0.85 are stranded in the upper-right: good quality, unacceptable latency. Context-aware at threshold=0.50 and Commercial Cloud both land inside the green zone.](./plots/article_figG.png)

*Every configuration we tested, plotted by latency vs. quality improvement. The green zone is where you want to be: latency under 150ms and quality improvement over our +15pp target. Three context-aware configurations are stranded at the right — great quality, unusable latency. The one that makes it into the spec-compliant zone is threshold=0.50. It's the only context-aware configuration where the AI judge never runs.*

Only two configurations land in the spec-compliant zone: our threshold=0.50 setup (zero AI judge calls) and the commercial cloud baseline (always routes to the bigger model, no routing logic at all). Every configuration that actually invokes the AI judge fails the latency spec.

---

## Why Does the Simple Rule Work?

This requires some explaining, because it seems counterintuitive. If a dumb rule works as well as a smart AI, what was the AI actually doing?

The answer: **the AI judge was trying to predict something that's already directly measurable.**

The context wall is deterministic. It hits when you run out of tokens. Token count is something you can measure exactly, right now, without any inference. It's not ambiguous or probabilistic. It's a number.

The AI judge was looking at conversation history, task complexity, linguistic signals — trying to estimate whether context saturation was imminent. But context saturation isn't hidden. It's already in the metrics. The judge was using complex inference to approximate a value that was available directly.

Here's the analogy: imagine using a GPS to predict whether you've driven more than 100 miles, when the odometer is sitting right in front of you showing the exact mileage. The GPS might give you a reasonable estimate. But the odometer gives you the ground truth in milliseconds, for free.

The completion wall is an odometer problem. Treating it as a GPS problem added complexity, latency, and noise — without improving accuracy.

---

## When AI Judges Help (And When They Don't)

This experiment clarified something we'd been fuzzy on: **LLM judges are valuable for unstructured, subjective, or multi-factor decisions — not for decisions where the signal is already structured and measurable.**

When does it make sense to use an AI to make a routing decision?

- The decision depends on meaning, not just metrics — "is this question off-topic for this agent?"
- Multiple signals need to be balanced — latency preference, cost budget, task complexity, user tier
- The ground truth isn't directly observable — "will this session require follow-up?"

When doesn't it?

- The decision variable is a number you already have — context occupancy, queue depth, token count
- The threshold is known and stable — not something that shifts based on context
- Speed matters — you're making this decision on every turn of every session

For our problem — "is context filling up?" — the signal was already there, precise, and cheap to check. Adding an AI layer made things slower and less reliable.

---

## What's Still Unsolved

Setting threshold=0.50 fixes the latency problem. It does not fix the reactivity problem.

At threshold=0.50, the router escalates when context hits 50% occupancy. That's better than escalating at 85% — you get more headroom. But it's still reactive: you're measuring the current state, not predicting the future trajectory.

A session that's at 50% on turn 2 of a long, complex task will probably hit the wall. A session at 50% on turn 18 of a quick exchange won't. The rule can't distinguish them.

**What would actually fix the reactivity problem is velocity-based routing**: don't measure where you are, measure how fast you're getting there. If context is growing at 400 tokens per turn on a task with 10 turns left, you can predict the collision — and escalate before you're anywhere near the threshold.

That's a forecasting problem. And forecasting — predicting future state from current trajectory — is actually a case where a smarter component might earn its latency cost. The structured occupancy check has hit its ceiling. What comes next needs to see the arc, not just the point.

That's the next experiment.

---

## What We Learned

Four takeaways from the threshold sweep:

**1. Run ablation experiments before assuming your complex component is doing useful work.** We assumed the AI judge was improving routing quality. The ablation showed it was adding latency without adding value. We would not have known this without testing configurations that disable it.

**2. For structured, measurable signals: use deterministic rules.** When you already have the number that matters, a model that approximates it adds noise, not intelligence.

**3. The Pareto frontier is the right way to visualize quality-latency tradeoffs.** A system that's 100% on quality and 70× over on latency isn't "good enough on quality" — it's not in the acceptable zone. Plotting both dimensions together makes this obvious in a way that separate metrics don't.

**4. Building AI systems often means discovering which parts don't need AI.** We built a sophisticated routing system. The best version of it is a single comparison operator. That's not a failure — that's what the experiment was for.

---

## The Series in Summary

We started with a real problem: local LLMs fail catastrophically when context gets long, and the failure is predictable but easy to miss.

We built a context-aware router to solve it. The quality improvement was real — +36 to +44 percentage points at high context depth. But the naive implementation was 70× too slow to use.

A threshold sweep revealed that the AI judge — the "smart" part of the system — was the source of both the latency and the imprecision. Disabling it by lowering the threshold to 0.50 preserved the quality gain while dropping worst-case latency from 10 seconds to 0.04 milliseconds.

The lesson isn't that AI is bad at routing. It's that the right tool for a deterministic threshold problem is a deterministic rule. And that finding out which parts of your system are actually necessary requires testing configurations that remove them.

---

*Addi Katzav — June 2026*

*[← Part 1](./article_1_the_cliff.md) · [← Part 2](./article_2_too_slow.md) · Part 3 of 3*
