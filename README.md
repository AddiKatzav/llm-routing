# LLM Routing Benchmark

Benchmarks three LLM routing paradigms (static semantic, context-aware, commercial cloud) in long-horizon agentic loops using local Ollama models.

See `routing_benchmark_spec.md` for the full experiment specification and `EXPERIMENT_REPORT.md` for baseline results.

## Prerequisites

- Python ≥ 3.10
- [Ollama](https://ollama.com) running locally

## Models

Pull the three models the benchmark uses:

```bash
ollama pull llama3.2:3b       # local model
ollama pull llama3.1:8b       # cloud stand-in (no real API key needed)
ollama pull qwen2.5:1.5b      # context-aware router judge
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

For Anthropic cloud routing experiments (optional):

```bash
pip install -e ".[cloud]"
export ANTHROPIC_API_KEY=sk-...
```

## Run experiments

```bash
# Quick smoke test (~6 tasks, finishes in minutes)
python scripts/run_benchmark.py --subset demo

# Full overnight sweep (~60 tasks × 3 routers)
python scripts/run_benchmark.py --subset overnight --n-repeats 1

# Threshold tuning sweep (vary when context-aware router escalates)
python scripts/run_benchmark.py --subset threshold_tuning --wall-threshold 0.65 --output-dir run_tuned_065
python scripts/run_benchmark.py --subset threshold_tuning --wall-threshold 0.70 --output-dir run_tuned_070
```

Results are written to `results/<output-dir>/` (turns.jsonl, runs.csv, kpi_summary.json).

Key flags: `--subset {demo,overnight,threshold_tuning}`, `--output-dir NAME`, `--n-repeats N`, `--wall-threshold FLOAT` (default 0.85).

## Analyze results

```bash
python scripts/analyze_results.py results/run_overnight
python scripts/generate_plots.py results/run_overnight
python scripts/article_plots.py   # publication figures
python scripts/article_analysis.py
```

## Tests

```bash
pytest
```

## Project layout

```
routing_benchmark/     Core library (models, routers, providers, metrics)
  routers/             StaticSemanticRouter, ContextAwareRouter, CommercialCloudRouter
  providers/           OllamaProvider, AnthropicCloudProvider
scripts/               Benchmark runner and analysis scripts
tests/                 Unit and integration tests (18 files)
results/               Benchmark run outputs — gitignored, regenerable
plots/                 Generated figures — gitignored, regenerable
```
