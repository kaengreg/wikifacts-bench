"""Print BEIR metrics for one adapted retrieval file.

The file is the JSON written by ``get_scores_batch.py``,
shaped as ``{query_id: {corpus_id: score}}``. ``--folder`` selects the qrels
that match the chunk type.

Example:

    python retrieval_pipeline/eval_retrieval_single.py \
      --folder sentences \
      --results heavy_artifacts/evaluation/sentences/e5/rrf-k60/results.json

The same text is printed and written to ``results_metrics.txt`` beside that JSON.
The headline values are one tab-separated row.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_retrieval import (
    FOLDER_TO_DATASET,
    HEADLINE_METRICS,
    REPO_ROOT,
    dataset_repos,
    load_data,
    score_results,
)
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval


def resolve_results_path(raw_path: str | None) -> Path:
    if raw_path is None:
        return REPO_ROOT / "heavy_artifacts" / "results_adapted.json"
    path = Path(raw_path)
    if path.is_absolute():
        return path
    from_cwd = Path.cwd() / path
    if from_cwd.exists():
        return from_cwd
    return REPO_ROOT / path


def metrics_output_path(results_path: Path, raw_output: str | None) -> Path:
    if raw_output is None:
        return results_path.with_name(f"{results_path.stem}_metrics.txt")
    path = Path(raw_output)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def format_metrics_report(results_path: Path, hf_repo: str, hf_repo_qrels: str, scored: dict) -> str:
    """Text report. The headline block is tab-separated so it pastes into a sheet."""
    lines = [
        f"Results: {results_path}",
        f"Corpus: {hf_repo}",
        f"Qrels: {hf_repo_qrels}",
        "",
        "Headline metrics:",
        "\t".join(label for label, _group, _key in HEADLINE_METRICS),
        "\t".join(str(scored[group].get(key, 0)) for _label, group, key in HEADLINE_METRICS),
        "",
    ]
    for label, group, key in HEADLINE_METRICS:
        lines.append(f"{label}\t{scored[group].get(key, 0)}")

    lines.append("")
    lines.append("All cutoffs:")
    for group in ("ndcg", "map", "recall", "precision", "mrr"):
        for metric_name, metric_value in scored[group].items():
            lines.append(f"{metric_name} {metric_value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Score one adapted retrieval JSON file and print the metrics.")
    parser.add_argument(
        "--folder",
        default="para",
        choices=sorted(FOLDER_TO_DATASET),
        help="Chunk type. Selects the WikiFacts corpus and qrels. Default matches the old paragraph run.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Adapted JSON file. Default: heavy_artifacts/results_adapted.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Metrics text file. Default: <results stem>_metrics.txt next to the JSON file.",
    )
    args = parser.parse_args()

    results_path = resolve_results_path(args.results)
    if not results_path.exists():
        parser.error(f"Results file not found: {results_path}")

    hf_repo, hf_repo_qrels = dataset_repos(args.folder)
    qrels = load_data(hf_repo, hf_repo_qrels)
    retriever = EvaluateRetrieval(k_values=[1, 3, 5, 10, 100])
    scored = score_results(results_path, qrels, retriever)

    report = format_metrics_report(results_path, hf_repo, hf_repo_qrels, scored)
    output_path = metrics_output_path(results_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(report, end="")
    print(f"Wrote metrics to {output_path}")


if __name__ == "__main__":
    main()
