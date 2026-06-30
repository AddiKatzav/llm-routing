# Your Local AI Has a Hidden Cliff

*Part 1 of 3 — LLM Routing in Practice*

---

It's working great. You're building an agent that reads files, runs commands, drafts code. Turn 3, turn 6, turn 9 — clean responses, exactly what you asked for. Then somewhere around turn 12, something shifts. The reply cuts off mid-sentence. The next one is a near-repeat of what came before. Then the model starts generating something that technically looks like an answer but doesn't mean anything at all.

You didn't change anything. Your prompt is fine. The model isn't broken. What happened?

You hit the cliff.

---

## Local LLMs Have a Hard Ceiling — Not a Soft One

Every language model has a **context window** — a limit on how much text it can hold in memory at once. For small local models like `llama3.2:3b`, that limit is 4,096 tokens. About 3,000 words. In a multi-turn agent session, that fills up faster than you expect: your initial prompt, each tool call, each response — it all accumulates.

The important thing to understand is what happens when you approach that limit. It's not a gradual slowdown. It's a cliff.

Think of RAM on an old computer. When you're at 70% memory usage, everything works fine. At 90%, still mostly fine. But when you run out? The system doesn't gently degrade — it thrashes, freezes, crashes. Local LLMs behave the same way around their context limit: performance holds up until it doesn't, and then it falls apart completely.

We call this the **completion wall** — the point where context saturation triggers catastrophic failure in generation quality. Responses truncate. Tool calls come back malformed. The model starts looping or produces nothing useful at all.

**The frustrating part: this is completely predictable.** You can watch the context fill up in real time. It's not random. It's physics.

---

## Why It's Easy to Miss

If you mostly test your agent on short conversations, you'll never see this. Single-turn or two-turn prompts never come close to the context limit. Even a 10-turn session with concise exchanges might stay well under 50% capacity.

The problem emerges in real-world usage patterns: debugging sessions that go deep, code reviews over large files, planning conversations that run long. Those are exactly the cases where you most need the agent to perform well — and they're the cases where the cliff is waiting.

By the time you notice the problem, you're already past the cliff. The session that was working fine is now producing garbage. You restart it, it works fine again for a while, and eventually hits the wall again. It looks like a fluke. It's not.

---

## The Obvious Fix — And Why It's Incomplete

The standard solution is **cloud escalation**: when a conversation gets long, hand it off to a larger model with a bigger context window. A cloud model like Claude or GPT-4 handles 100,000+ tokens. The cliff essentially disappears.

But this creates its own tradeoffs:

- **Cost:** Cloud API calls add up fast at scale. Running everything through cloud from the start destroys the economics of using local models at all.
- **Latency:** Cloud inference has real network overhead. For interactive agents where responsiveness matters, that latency is felt by the user.
- **Privacy:** If your agent is working with sensitive data, you may not want it leaving your machine at all.

The goal is to use local models as much as possible — and escalate to cloud only when necessary. That means you need a way to decide *when* to escalate. A **router**.

---

## Two Approaches to Routing

The simplest approach is a **Static Semantic** router: at the start of a session, it compares the initial prompt against a library of task examples using embedding similarity. If the match suggests the task is complex or likely to run long, it routes to cloud. Otherwise it stays local. One decision, cached for the entire session.

The problem: a Static Semantic router can't see the future. It has to guess, based on the initial prompt, whether context will become a problem by turn 15. Sometimes it guesses right. Often it doesn't.

A smarter approach is **dynamic routing**: re-evaluate on every turn. Look at the current context state — how full is the window? — and decide each time whether to stay local or escalate. If you're running out of room, escalate. If you have plenty of space, stay local.

This sounds like an obvious improvement. We built it and ran a real benchmark to find out if it actually works.

---

## What We Measured

We ran 200 benchmark conversations: 100 with a static router, 100 with a context-aware dynamic router. Each session was a multi-turn agentic task — the kind of work a coding or data-analysis agent might do.

To make the comparison clean and focused, we controlled the starting context depth. We ran sessions at four different levels:

| Depth | Context fill at start | What it tests |
|-------|----------------------|---------------|
| Shallow | ~10% | Normal, comfortable sessions |
| Mid | ~50% | Getting full, but not urgent |
| Near wall | ~85% | Close to the edge |
| Over wall | >100% | Already past the limit |

The "near wall" and "over wall" conditions are where the cliff lives. We wanted to measure exactly how much each router helps in those situations.

Our success metric was simple: **did the agent finish the task without hitting the wall?** We called this the wall-avoidance score.

---

## The Result

![Wall-avoidance score at four context depth levels. Static Semantic routing (blue) and context-aware routing (green) perform identically at shallow and mid depth. At near-wall and over-wall, context-aware routing achieves 100% while Static Semantic routing trails by 36–44 percentage points. Commercial Cloud (dashed grey) is the quality ceiling across all depths.](./plots/article1_wall_avoidance.png)

*At shallow and mid depth, all three approaches perform identically — there's no cliff to avoid. The gap opens sharply at near-wall (+36 percentage points) and over-wall (+44 percentage points). Context-aware routing avoids the cliff on every single session. The Static Semantic router fails on more than a third. Commercial Cloud is the quality ceiling: immune to context wall failures by design.*

At normal context depths, routing strategy doesn't matter — there's no imminent cliff. But as sessions get long and context fills up, the dynamic router pulls far ahead.

At near-wall depth: the context-aware router avoided the cliff on **100% of sessions**. The Static Semantic router: **64%**. A gap of **+36 percentage points**. At over-wall depth, the gap widens to **+44 points**.

That's a meaningful result. The routing approach met — and significantly exceeded — our quality target.

So: context-aware routing works. The improvement is real.

But building it revealed a much nastier problem. One we didn't anticipate.

---

## What's Next

In Part 2, we look at what happens when you measure not just quality, but latency. The context-aware router's quality numbers were great. Its worst-case response time was 10,437 milliseconds. The target was 150ms.

We were 70× over budget.

[→ Part 2: We Built a Smart Fix. It Worked. It Was Also 70× Too Slow.](./article_2_too_slow.md)

---

*Addi Katzav — June 2026*

*Part 1 of 3. [Part 2](./article_2_too_slow.md) · [Part 3](./article_3_the_if_statement.md)*
