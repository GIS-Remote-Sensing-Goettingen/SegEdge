# Agent Playbook for SegEdge

This document outlines how to collaborate with SegEdge using agentic LLM tooling (e.g. GPT‑5 Codex). Treat it as the single source of truth when delegating tasks to autonomous coding agents.

---

## 1. Pre‑Flight Checklist
- **Project install**: Always run `pip install -e .[dinov3,sam2]` (or the minimal extra required for the task) before executing pipelines. This ensures the `sege` package and optional dependencies resolve without manual `PYTHONPATH` hacks.
- **Environment parity**: When staging experiments, mirror the repo directory layout (`src/`, `experiments/`, `artifacts/`, `data/`). Never write outside these roots unless explicitly requested.
- **Model assets**: Verify DINOv3 and SAM 2 weights are present under `artifacts/checkpoints/`. If `TRANSFORMERS_OFFLINE=1` is set, fail early with a clear message if weights are missing.
- **Sample data**: Use imagery in `data/samples/imagery/` for smoke tests. Do not bundle large proprietary datasets; reference them via arguments or environment variables.

---

## 2. Code Quality Standards
- **Module pattern**: New pipelines must live under `src/sege/pipelines/`, expose a `main(argv: list[str] | None = None)` entry point, and default to repository-relative paths (see `dinov3_tree_crowns.py` lines 1‑60 for reference).
- **Typing & imports**: Prefer explicit typing (`Path`, `list[str]`) and maintain the existing import grouping (stdlib → third-party → local). Avoid `from … import *`.
- **Configuration**: Extend `pyproject.toml` optional dependencies when your code introduces new libraries. Keep `requirements/experiments/*.txt` in sync if agents rely on pip requirements files.
- **Design symmetry**: Reuse existing utilities (e.g. `StageProfiler`, tiling helpers) instead of reimplementing. When adding new helpers, place them in `src/sege/utils/` with minimal surface area and documentation.
- **Defensive fallbacks**: Mirror existing fallbacks (e.g. CPU inference when CUDA is unavailable, alternate clustering heuristics). Always log fallbacks with actionable guidance.

---

## 3. Logging & Observability
- **Structured logging**: Use named loggers (`logging.getLogger("module_name")`) and maintain the timestamped format configured in `setup_logging` inside `dinov3_tree_crowns.py`.
- **Profiler integration**: Wrap expensive blocks with `StageProfiler.track("label")` so they appear in the summary report. Record new stages near existing ones (profiling output is part of the QA workflow).
- **CUDA telemetry**: Call `log_cuda("message")` after heavy GPU operations. If adding new GPU calls, import and reuse the helper rather than printing raw `torch.cuda` stats.
- **HPC feedback**: Ensure SLURM jobs echo environment information (`python --version`, `torch.utils.collect_env`). Follow the pattern in `experiments/dinov3/hpc/slurm_dinov3.sh`.
- **Failure visibility**: When raising errors (missing files, unsupported shapes), include remediation hints. Example: `"Head checkpoint missing. Run the linear probe trainer or switch to --mode unsup."`

---

## 4. Benchmarking & Validation
- **Static verification**: Always run `python -m compileall src/sege` (and any changed experiment scripts) to catch syntax errors. Agents must report the command result in their final summary.
- **Functional smoke tests**:
  - DINOv3: `PYTHONPATH=$PWD/src python -m sege.pipelines.dinov3_tree_crowns --input data/samples/imagery/dinov3_smoll.tif --mode unsup --device cpu`
  - SAM 2: `PYTHONPATH=$PWD/src python -m sege.pipelines.sam2_farmland --input data/samples/imagery/1084-1393.tif --device cpu --tile-size 0`
  Only skip when dependencies are unavailable, and explicitly state the reason.
- **Performance tracking**: If changes affect runtime, capture before/after metrics using the profiler summary or external benchmarking tools. Store benchmark notes in `docs/operations/` (create a dated Markdown report if one doesn’t exist).
- **Regression safety**: Never delete sample outputs under `artifacts/outputs/<pipeline>/` without replacing them. Agents should generate fresh artifacts when changing algorithms to support manual visual QA.

---

## 5. Documentation & Communication
- **Markdown standards**: Follow the concise, bullet-first style used across `docs/pipelines/*.md`. Include requirements, inputs, pipeline stages, outputs, failure modes, and references.
- **Agents log**: When agents perform non-trivial maintenance, append a dated entry to `docs/operations/CHANGELOG.md` (create the file if needed) summarizing intent, approach, and validation.
- **README alignment**: Update `README.md` whenever the high-level workflow, installation steps, or layout changes. Keep the table in sync with new pipelines.
- **Inline comments**: Add comments sparingly for complex logic or heuristics (e.g. cluster scoring, prompt selection). Avoid restating obvious operations.
- **Checkpoint hygiene**: Document any new required external assets in `docs/references/` with download instructions and licensing notes.

---

## 6. Design Patterns to Follow
- **Repository-relative defaults**: Derive file paths via `PROJECT_ROOT = Path(__file__).resolve().parents[3]` and use subdirectories under `artifacts/` and `data/`.
- **Config-driven HPC**: Thin SLURM wrappers should only load modules, export `PYTHONPATH`, and call a configuration-aware driver (see `experiments/dinov3/hpc/tiling_tree_crowns.py`).
- **CLI duality**: Every executable script must accept CLI arguments and allow programmatic invocation via function call (e.g. `dinov3_tree_crowns.main(argv)`).
- **Fallback heuristics**: When decisions rely on heuristics (cluster scoring, vegetation prompts), log the metrics and provide alternative execution paths (e.g. alternative clusters, relaxed thresholds).
- **Vendor separation**: Upstream repos stay in `third_party/`. When touching vendor code, prefer adapter layers in `src/sege/utils/` or patch files using declarative diffs to keep updates manageable.

---

## 7. Common Pitfalls & Guardrails
- **Missing extras**: If a command raises `ModuleNotFoundError`, first ensure the relevant optional extra has been installed. Agents must never hardcode `sys.path` hacks in production scripts.
- **Weights download**: When offline weights are missing, emit a targeted error instead of silently falling back to online downloads. Reference the exact Hugging Face model ID in the message.
- **Large file churn**: Never commit generated `.tif`, `.gpkg`, or model weights. `.gitignore` already covers these; agents should verify no large artifacts slip through using `git status`.
- **Notebook edits**: Keep notebooks within `notebooks/` or `experiments/<pipeline>/prototyping/`. If an agent modifies a notebook, they must clear outputs before saving.
- **Concurrency**: If implementing parallel inference, guard CUDA calls with synchronization and update logging to signal concurrency mode.

---

By adhering to this playbook, agentic assistants will produce changes that mesh with SegEdge’s conventions, remain observable in production, and preserve the system-level architecture already in place. Treat deviations as exceptions requiring explicit approval.***
