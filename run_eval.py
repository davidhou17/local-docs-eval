#!/usr/bin/env python3
"""
Docs Q&A golden test set evaluation.

Loads a CSV of (Question_ID, Category, Question, Answer), runs each question
through one or more Ollama models with RAG over local docs, scores the response
with an LLM-as-judge (1 / 0.75 / 0), and reports mean documentation accuracy.

Usage:
  python -m docs_eval --docs-dir ../docs --models llama3.2
  python -m docs_eval --docs-dir ../docs --models llama3.2,mistral --judge-model llama3.2
  python -m docs_eval --docs-dir ../docs --workers 4 --no-think --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .judge import JudgeResult, judge
from .responders import get_responder


_DEFAULT_CSV = str(Path(__file__).resolve().parent / "golden-dataset-docs-q-and-a.csv")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    """Load golden dataset CSV; columns: Question_ID, Category, Question, Answer."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Checkpoint helpers  (only active when --output is set)
# ---------------------------------------------------------------------------

def _checkpoint_path(output_path: str) -> str:
    p = Path(output_path)
    return str(p.with_suffix(".ckpt.jsonl"))


def _load_checkpoint(ckpt_path: str) -> tuple[set[tuple[str, str]], list[dict]]:
    """Return (done_set, results) from an existing checkpoint file."""
    done: set[tuple[str, str]] = set()
    results: list[dict] = []
    if not os.path.exists(ckpt_path):
        return done, results
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["model"], r["Question_ID"]))
                results.append(r)
            except (json.JSONDecodeError, KeyError):
                pass
    return done, results


def _append_checkpoint(ckpt_path: str, result: dict, lock: threading.Lock) -> None:
    with lock:
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_single_model(
    rows: list[dict],
    model: str,
    judge_model: str,
    responder_name: str,
    responder_kwargs: dict,
    judge_kwargs: dict | None = None,
    workers: int = 1,
    ckpt_path: str | None = None,
    ckpt_lock: threading.Lock | None = None,
) -> list[dict]:
    """Evaluate one model on all rows. Returns list of per-row result dicts."""
    if not rows:
        return []

    responder = get_responder(responder_name, model=model, **responder_kwargs)
    _judge_kwargs = judge_kwargs or {}
    total = len(rows)
    results: list[dict] = []
    results_lock = threading.Lock()

    def eval_one(row: dict) -> dict:
        question_id = row.get("Question_ID", "")
        category = row.get("Category", "")
        question = row.get("Question", "")
        golden_answer = row.get("Answer", "")

        try:
            model_response = responder.get_response(
                question_id=question_id,
                question=question,
                category=category,
                golden_answer=golden_answer,
            )
            judge_result: JudgeResult = judge(
                question=question,
                expected_answer=golden_answer,
                model_response=model_response,
                model=judge_model,
                **_judge_kwargs,
            )
            result = {
                "model": model,
                "Question_ID": question_id,
                "Category": category,
                "Question": question[:200] + "..." if len(question) > 200 else question,
                "score": judge_result.score,
                "reasoning": judge_result.reasoning,
                "model_response": (model_response[:500] + "...") if len(model_response) > 500 else model_response,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "model": model,
                "Question_ID": question_id,
                "Category": category,
                "Question": question[:200] + "..." if len(question) > 200 else question,
                "score": None,
                "reasoning": "",
                "model_response": "",
                "error": str(exc),
            }

        if ckpt_path and ckpt_lock:
            _append_checkpoint(ckpt_path, result, ckpt_lock)

        return result

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(eval_one, row) for row in rows]
        for done_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with results_lock:
                results.append(result)
                error_count = sum(1 for r in results if r.get("error"))

            elapsed = time.monotonic() - start
            rate = done_count / elapsed if elapsed > 0 else 0
            eta_s = int((total - done_count) / rate) if rate > 0 else 0
            eta_str = f" ETA {eta_s}s" if eta_s > 0 else ""
            err_str = f" | errors: {error_count}" if error_count else ""
            score_str = f"score={result['score']}" if result["score"] is not None else f"ERROR: {result['error'][:60]}"
            print(
                f"  [{done_count}/{total}{err_str}]{eta_str} Q{result['Question_ID']} {score_str}   ",
                end="\r",
                flush=True,
            )

    print()  # clear the \r line
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(model: str, results: list[dict]) -> None:
    """Print per-model summary: mean score, distribution, by-category breakdown."""
    valid = [r for r in results if r.get("score") is not None]
    errors = [r for r in results if r.get("error")]
    scores = [r["score"] for r in valid]

    mean_score = sum(scores) / len(scores) if scores else 0.0
    count_1 = sum(1 for s in scores if s == 1.0)
    count_075 = sum(1 for s in scores if s == 0.75)
    count_0 = sum(1 for s in scores if s == 0.0)

    print(f"\n--- {model} ---")
    print(f"  Mean accuracy: {mean_score:.3f}  (n={len(valid)})")
    print(f"  Distribution:  1.0={count_1}  0.75={count_075}  0.0={count_0}")

    if errors:
        print(f"  Errors: {len(errors)} question(s) failed (excluded from scoring)")
        for r in errors[:5]:
            print(f"    Q{r['Question_ID']}: {(r['error'] or '')[:120]}")

    by_category: dict[str, list[float]] = {}
    for r in valid:
        by_category.setdefault(r["Category"], []).append(r["score"])

    if by_category:
        print("  By category:")
        for cat in sorted(by_category.keys()):
            cat_scores = by_category[cat]
            cat_mean = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0
            print(f"    {cat}: {cat_mean:.3f} (n={len(cat_scores)})")


def print_comparison_table(all_results: dict[str, list[dict]]) -> None:
    """Print a side-by-side comparison table when multiple models are evaluated."""
    if len(all_results) < 2:
        return

    models = list(all_results.keys())

    def model_mean(rs: list[dict]) -> float:
        valid = [r["score"] for r in rs if r.get("score") is not None]
        return sum(valid) / len(valid) if valid else 0.0

    means = {m: model_mean(rs) for m, rs in all_results.items()}

    categories: set[str] = set()
    for rs in all_results.values():
        for r in rs:
            categories.add(r["Category"])

    cat_means: dict[str, dict[str, float]] = {}
    for cat in sorted(categories):
        cat_means[cat] = {}
        for m, rs in all_results.items():
            cat_scores = [r["score"] for r in rs if r["Category"] == cat and r.get("score") is not None]
            cat_means[cat][m] = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0

    col_width = max(len(m) for m in models) + 2
    cat_width = max((len(c) for c in categories), default=10) + 2
    cat_width = max(cat_width, len("Category") + 2)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    header = f"{'':>{cat_width}}" + "".join(f"{m:>{col_width}}" for m in models)
    print(header)
    print("-" * len(header))

    overall_row = f"{'OVERALL':>{cat_width}}" + "".join(f"{means[m]:>{col_width}.3f}" for m in models)
    print(overall_row)
    print("-" * len(header))

    for cat in sorted(categories):
        row = f"{cat:>{cat_width}}" + "".join(f"{cat_means[cat][m]:>{col_width}.3f}" for m in models)
        print(row)

    print("=" * 60)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_eval(
    csv_path: str,
    models: list[str],
    judge_model: str,
    responder_name: str,
    responder_kwargs: dict,
    judge_kwargs: dict | None = None,
    limit: int | None = None,
    output_path: str | None = None,
    workers: int = 1,
    use_checkpoint: bool = True,
) -> None:
    """Run evaluation for one or more models, aggregate results, optionally write CSV."""
    rows = load_csv(csv_path)
    if limit is not None:
        rows = rows[:limit]

    # Checkpoint setup
    ckpt_path: str | None = None
    ckpt_done: set[tuple[str, str]] = set()
    ckpt_results: list[dict] = []
    ckpt_lock = threading.Lock()

    if output_path and use_checkpoint:
        ckpt_path = _checkpoint_path(output_path)
        ckpt_done, ckpt_results = _load_checkpoint(ckpt_path)
        if ckpt_done:
            print(f"Resuming from checkpoint: {len(ckpt_done)} question(s) already completed.")

    all_results: dict[str, list[dict]] = {}

    for model in models:
        preloaded = [r for r in ckpt_results if r["model"] == model]
        model_done_ids = {qid for (m, qid) in ckpt_done if m == model}
        remaining = [row for row in rows if row.get("Question_ID", "") not in model_done_ids]

        print(f"\nEvaluating model: {model}  (judge: {judge_model}, responder: {responder_name})")
        if preloaded:
            print(f"  Skipping {len(preloaded)} already-checkpointed, running {len(remaining)} remaining.")
        print("-" * 50)

        new_results = run_single_model(
            remaining,
            model=model,
            judge_model=judge_model,
            responder_name=responder_name,
            responder_kwargs=responder_kwargs,
            judge_kwargs=judge_kwargs,
            workers=workers,
            ckpt_path=ckpt_path,
            ckpt_lock=ckpt_lock,
        )

        all_results[model] = preloaded + new_results
        print_summary(model, all_results[model])

    print_comparison_table(all_results)

    if output_path:
        combined = [r for rs in all_results.values() for r in rs]
        fieldnames = ["model", "Question_ID", "Category", "Question", "score", "reasoning", "model_response", "error"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(combined)
        print(f"\nWrote per-row results to {output_path}")

        # Clean up checkpoint on successful completion
        if ckpt_path and os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"Checkpoint removed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Docs Q&A golden test set evaluation (Ollama)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=os.environ.get("GOLDEN_DATASET_CSV", _DEFAULT_CSV),
        help="Path to golden dataset CSV (columns: Question_ID, Category, Question, Answer).",
    )
    parser.add_argument(
        "--models",
        default="qwen3:8b",
        help="Comma-separated Ollama model names to evaluate (e.g. qwen3:8b,gemma3:12b,llama3.3).",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3:8b",
        help="Ollama model to use as the LLM judge.",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Path to docs directory containing .mdx files for RAG. Required for 'rag' responder.",
    )
    parser.add_argument(
        "--embed-model",
        default="mxbai-embed-large",
        help="Ollama embedding model for RAG retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of doc chunks to retrieve per question.",
    )
    parser.add_argument(
        "--responder",
        default="rag",
        choices=["rag", "naive_baseline", "golden_answer"],
        help="Responder type. 'naive_baseline' skips RAG and tests raw LLM knowledge.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for question evaluation. >1 sends concurrent requests to Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-call timeout in seconds for all LLM calls (responder and judge).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient Ollama errors (network errors, HTTP 5xx).",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help="Disable internal chain-of-thought for all LLM calls. "
             "Recommended when using thinking models (e.g. qwen3) to prevent token budget exhaustion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write per-row results CSV. Also enables checkpoint/resume.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        default=False,
        help="Disable checkpoint/resume even when --output is set.",
    )
    args = parser.parse_args()

    if not args.csv or not os.path.isfile(args.csv):
        print(f"Error: CSV not found at '{args.csv}'.", file=sys.stderr)
        print("Provide --csv /path/to/golden-dataset.csv or set GOLDEN_DATASET_CSV.", file=sys.stderr)
        sys.exit(1)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        print("Error: --models requires at least one model name.", file=sys.stderr)
        sys.exit(1)

    responder_kwargs: dict = {
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    if args.no_think:
        responder_kwargs["think"] = False

    if args.responder == "rag":
        if not args.docs_dir:
            auto_docs = Path(__file__).resolve().parent.parent / "docs"
            if auto_docs.is_dir():
                args.docs_dir = str(auto_docs)
            else:
                print("Error: --docs-dir is required for the 'rag' responder.", file=sys.stderr)
                print("Point it at your docs directory containing .mdx files.", file=sys.stderr)
                sys.exit(1)

        if not os.path.isdir(args.docs_dir):
            print(f"Error: docs directory not found at '{args.docs_dir}'.", file=sys.stderr)
            sys.exit(1)

        responder_kwargs["docs_dir"] = args.docs_dir
        responder_kwargs["embed_model"] = args.embed_model
        responder_kwargs["top_k"] = args.top_k

    judge_kwargs: dict = {
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }

    run_eval(
        csv_path=args.csv,
        models=model_list,
        judge_model=args.judge_model,
        responder_name=args.responder,
        responder_kwargs=responder_kwargs,
        judge_kwargs=judge_kwargs,
        limit=args.limit,
        output_path=args.output,
        workers=args.workers,
        use_checkpoint=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
