# We Built a Smart Fix. It Worked. It Was Also 70× Too Slow.

*Part 2 of 3 — LLM Routing in Practice*

*Previously: [Part 1 — Your Local AI Has a Hidden Cliff](./article_1_the_cliff.md)*

---

The numbers looked perfect.

Context-aware routing improved our wall-avoidance score by **+36 percentage points** at near-wall depth. Every single session avoided the cliff. The Static Semantic router — the baseline we were trying to beat — failed on more than a third of sessions in the same conditions.

By any quality metric, we had a working system. We were confident. We wrote up the result.

Then we looked at the latency graph.

---

## How the Router Works (And Where It Goes Wrong)

Before we get to the numbers, it helps to understand the architecture of the context-aware router we built. It has two decision paths, which is where the trouble starts.

**Path 1 — The rule path:** Check a single number: how full is the context window right now? If it's above a threshold (we set it at 85% full), escalate to cloud immediately. This is pure arithmetic. Takes less than 1 millisecond.

**Path 2 — The judge path:** If the context isn't past the threshold yet, ask a small AI model (`qwen2.5:1.5b`) to evaluate the current conversation state and decide whether to escalate. The judge reasons about complexity, task trajectory, and conversational signals. It returns a one-word verdict: LOCAL or CLOUD.

The judge path is more nuanced. It can catch cases the rule path misses — a session at 50% occupancy that's about to blow up because the remaining task is very complex. That's the smart part.

The problem is that the smart part is also the slow part.

---

## The Number That Breaks Everything

The judge, running on CPU, takes anywhere from 50 milliseconds to 24 seconds per call. That's not a typo. Local CPU inference with a 1.5 billion parameter model is variable and sometimes very slow.

Our latency target was **150 milliseconds**. A router that takes more than 150ms to make a routing decision is adding noticeable lag to every agent turn.

Our actual worst-case routing latency: **10,437 milliseconds**.

That's 10.4 seconds. For a single routing decision. On a single agent turn.

| Router | Worst-case routing delay | Over target by |
|--------|--------------------------|----------------|
| Static Semantic (baseline) | 0.7 ms | — |
| Context-aware (our system) | 10,437 ms | **70×** |
| Target | 150 ms | — |

The system that worked perfectly on quality was completely unusable in practice.

---

## Why Is It So Slow?

The answer comes down to how often the judge gets called.

At our default threshold of 85% context occupancy, the rule path only fires when a session is already nearly full. Most sessions — especially at mid-depth where context starts around 50% — never reach that threshold. So they always fall through to the judge.

Result: **73% of all routing decisions invoke the LLM judge.**

![Routing latency distribution for threshold=0.50 (green) vs threshold=0.85 (red). The green line is a near-vertical spike — 100% of decisions complete in under 1ms. The red line has a fast early segment covering the 27% of rule-path decisions, then a heavy tail stretching to 24 seconds. The 150ms spec line sits in the flat middle of the red curve.](./plots/article_figD.png)

*Two routing configurations, same system. The green line (threshold set at 50% occupancy) shows 100% of decisions completing in under 1ms — essentially instant. The red line (threshold at 85%) shows 27% of fast decisions, then a heavy tail that extends to 24 seconds. The dashed line is our 150ms spec. Nearly three-quarters of all decisions in the red configuration blow past it.*

Think of it like a security checkpoint with two lanes. The fast lane uses an automatic scanner — scan your badge, door opens in milliseconds. The slow lane requires a human to review each person individually, which takes anywhere from 30 seconds to several minutes.

At threshold=0.85, 73% of passengers get sent to the slow lane — including plenty of people who clearly don't need a detailed review. The result is a massive bottleneck that makes the entire checkpoint unusable, even though the fast lane itself works perfectly.

---

## The Quality Numbers Are Misleading

Here's what makes this particularly frustrating: if you only measure task outcomes, the system looks great.

We measured Wall Avoidance Rate — did the agent complete the task without hitting the context limit? The context-aware router scored 100%. That's the right answer.

But "did it avoid the wall" and "was it usable" are different questions. A system that takes 10 seconds to make each routing decision would be rejected immediately in any real application. We were measuring the right thing for the research question, but the wrong thing for the product question.

This is a common trap: **optimizing what you measure while missing what matters.**

---

## It Also Escalates Too Late

While we were analyzing the latency data, we discovered a second problem with the escalation behavior — one that's separate from the speed issue but equally important.

We used something called shadow evaluation: for every turn the router sent to the cloud model, we also recorded what the local model would have produced. This let us label each cloud escalation as either "necessary" (the local model would have failed) or "unnecessary" (the local model would have been fine).

The result was surprising.

![Histogram showing when cloud escalations occur relative to the context limit. The x-axis shows how much context headroom remained at the moment of escalation. Positive values (green bars) mean escalation happened before the limit was reached. Negative values (red bars) mean escalation happened after the limit was already exceeded. Red bars dominate heavily, with the mean at -2.08.](./plots/article_figF.png)

*Each bar represents a cloud escalation. Green bars = the escalation happened before the context wall was hit (proactive). Red bars = the context wall was already exceeded at the moment of escalation (reactive). The average escalation fires when context is already more than double its limit. Only about 10% of escalations were proactive.*

**90% of cloud escalations fired after the context wall was already breached.** The router wasn't catching the problem before it happened — it was reacting after things had already gone wrong.

The timing problem makes sense once you see it: the rule `if occupancy ≥ 0.85, escalate` measures where you are *right now*, not where you're heading. By the time occupancy reaches 0.85, the model has already been degrading for several turns. The signal that triggers escalation lags the actual failure.

And **escalation precision was only 21%**: of 62 cloud escalations, only 13 were actually necessary. The other 49 were wasted — the local model would have been fine.

So the system was simultaneously: **too slow to use, usually too late, and wrong 79% of the time on which escalations were necessary.** But the WAR score was perfect.

This is why multiple measurement dimensions matter. A single headline metric can hide a lot.

---

## What We Learned

Four things to take away from this:

**1. Latency is a product requirement, not an implementation detail.** Measure it from the start of the experiment, not after you've celebrated the quality result. We should have instrumented routing overhead in our very first benchmark run.

**2. A system that's right on quality and wrong on latency isn't a working system.** In the real world, a 10-second routing delay isn't a performance issue — it's a user experience failure that makes the feature unusable.

**3. "Smart" doesn't automatically mean better.** Our AI-powered judge added latency, often fired too late, and escalated unnecessarily in 79% of cases. The dumb rule path — pure arithmetic — was faster, more precise, and in many ways more reliable.

**4. Shadow evaluation reveals problems invisible to outcome metrics.** Recording what would have happened on every turn, even when the router sent those turns to cloud, gave us ground truth we couldn't get any other way. That's how we found the 90% reactivity problem. If we'd only looked at WAR, we'd have missed it entirely.

---

## What's Next

We could have declared the system "good enough on quality" and moved on. Instead, we asked a different question: what happens if we turn the sensitivity dial all the way down?

The answer was one of the more surprising results of the whole experiment.

[→ Part 3: A Simple If-Statement Beat Our AI System](./article_3_the_if_statement.md)

---

*Addi Katzav — June 2026*

*[← Part 1](./article_1_the_cliff.md) · Part 2 of 3 · [Part 3 →](./article_3_the_if_statement.md)*
