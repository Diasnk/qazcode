"""
Small evaluation script: runs on the first 40 protocols only (for quick sanity checks).
Same CLI and behavior as evaluate.py, but limits to 40 protocols.
"""
import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# Logging for debugging evaluation issues
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("small_evaluate")
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Limit for small evaluation (first N protocols)
SMALL_EVAL_LIMIT = 5


@dataclass
class EvaluationResult:
    protocol_id: str
    accuracy_at_1: int  # 1 or 0
    recall_at_3: int  # 1 or 0
    latency_s: float
    ground_truth: str
    top_prediction: str
    top_3_predictions: list[str]
    response_json: dict


def _normalize_protocol_data(data: dict, json_file: Path) -> tuple[str, str, str, set]:
    """
    Extract protocol_id, query, ground_truth, and valid_icd_codes from a protocol JSON.
    Supports both full eval format (protocol_id, query, gt, icd_codes) and
    extracted_data format (identified_symptoms, gt only).
    """
    protocol_id = data.get("protocol_id") or json_file.stem
    ground_truth = data.get("gt")
    if ground_truth is None:
        raise ValueError(f"Missing 'gt' in {json_file.name}")

    # Query: prefer "query", else join "identified_symptoms"
    query = data.get("query") or ""
    if not query and data.get("identified_symptoms"):
        query = "\n".join(data["identified_symptoms"])
    query = (query or "").strip()

    # Valid ICD codes: prefer "icd_codes", else only gt (Recall@3 then = Accuracy@1 for that protocol)
    raw_codes = data.get("icd_codes")
    if raw_codes is not None:
        valid_icd_codes = set(raw_codes) if isinstance(raw_codes, (list, tuple)) else {raw_codes}
    else:
        valid_icd_codes = {ground_truth}
        logger.debug(
            "%s: no 'icd_codes', using gt only for valid set: %s",
            json_file.name,
            valid_icd_codes,
        )

    if ground_truth not in valid_icd_codes:
        raise ValueError(
            f"Dataset error in {json_file.name}: gt '{ground_truth}' not in icd_codes {valid_icd_codes!r}"
        )
    return protocol_id, query, ground_truth, valid_icd_codes


async def evaluate_single(
    client: httpx.AsyncClient,
    endpoint: str,
    json_file: Path,
    semaphore: asyncio.Semaphore,
) -> EvaluationResult:
    """Evaluate a single protocol against the endpoint."""
    async with semaphore:
        with open(json_file, "r") as f:
            data = json.load(f)

        protocol_id, query, ground_truth, valid_icd_codes = _normalize_protocol_data(
            data, json_file
        )

        logger.debug(
            "Evaluating %s: query_len=%d, gt=%s",
            json_file.name,
            len(query),
            ground_truth,
        )

        start_time = time.perf_counter()
        response = await client.post(endpoint, json={"symptoms": query})
        latency_s = time.perf_counter() - start_time

        response.raise_for_status()
        result = response.json()

        if "diagnoses" not in result:
            raise ValueError(
                f"Response missing 'diagnoses' for {json_file.name}. Keys: {list(result.keys())!r}"
            )
        diagnoses = sorted(result["diagnoses"], key=lambda x: x["rank"])
        top_3 = diagnoses[:3]

        top_prediction = diagnoses[0]["icd10_code"] if diagnoses else ""
        top_3_predictions = [d["icd10_code"] for d in top_3]

        # Accuracy@1: does the rank 1 prediction match ground truth?
        accuracy_at_1 = 1 if top_prediction == ground_truth else 0

        # Recall@3: are any of the top 3 predictions in the valid icd_codes list?
        recall_at_3 = (
            1 if any(code in valid_icd_codes for code in top_3_predictions) else 0
        )

        return EvaluationResult(
            protocol_id=protocol_id,
            accuracy_at_1=accuracy_at_1,
            recall_at_3=recall_at_3,
            latency_s=latency_s,
            ground_truth=ground_truth,
            top_prediction=top_prediction,
            top_3_predictions=top_3_predictions,
            response_json=result,
        )


async def run_evaluation(
    endpoint: str,
    dataset_dir: Path,
    parallelism: int,
    limit: int = SMALL_EVAL_LIMIT,
) -> list[EvaluationResult]:
    """Run evaluation on the first `limit` JSON files in the dataset directory."""
    console = Console()

    all_files = list(dataset_dir.glob("*.json"))
    if not all_files:
        console.print(f"[red]No JSON files found in {dataset_dir}[/red]")
        return []

    json_files = sorted(all_files)[:limit]
    total_in_dataset = len(all_files)

    console.print(
        Panel(
            f"[bold cyan]Diagnostic Accuracy Evaluation (small)[/bold cyan]\n\n"
            f"Endpoint: [yellow]{endpoint}[/yellow]\n"
            f"Dataset: [yellow]{dataset_dir}[/yellow]\n"
            f"Files: [yellow]{len(json_files)}[/yellow] of {total_in_dataset}\n"
            f"Parallelism: [yellow]{parallelism}[/yellow]",
            title="[bold white]Configuration[/bold white]",
            border_style="cyan",
        )
    )

    semaphore = asyncio.Semaphore(parallelism)
    results: list[EvaluationResult] = []
    errors: list[tuple[Path, Exception]] = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Evaluating protocols...", total=len(json_files)
            )

            async def process_file(json_file: Path):
                try:
                    result = await evaluate_single(
                        client, endpoint, json_file, semaphore
                    )
                    results.append(result)
                except Exception as e:
                    errors.append((json_file, e))
                    err_msg = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "Error evaluating %s: %s",
                        json_file.name,
                        err_msg,
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                finally:
                    progress.advance(task)

            await asyncio.gather(*[process_file(f) for f in json_files])

    if errors:
        console.print(
            f"\n[red]Encountered {len(errors)} errors during evaluation[/red]"
        )
        for path, err in errors[:5]:
            err_display = f"{type(err).__name__}: {err}" if str(err).strip() else f"{type(err).__name__}"
            console.print(f"  [dim]• {path.name}: {err_display}[/dim]")
        if len(errors) > 5:
            console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")
        console.print(
            "[dim]Run with -v/--verbose to see full tracebacks in logs.[/dim]"
        )

    return results


def compute_metrics(results: list[EvaluationResult]) -> dict:
    """Compute aggregated metrics from evaluation results."""
    if not results:
        return {}

    total = len(results)
    accuracy_at_1 = sum(r.accuracy_at_1 for r in results) / total * 100
    recall_at_3 = sum(r.recall_at_3 for r in results) / total * 100
    latencies = [r.latency_s for r in results]
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50_latency = statistics.median(latencies)

    if total >= 4:
        quantiles = statistics.quantiles(latencies, n=20)
        p95_latency = quantiles[-1]
    else:
        p95_latency = max_latency

    return {
        "total_protocols": total,
        "accuracy_at_1_percent": round(accuracy_at_1, 2),
        "recall_at_3_percent": round(recall_at_3, 2),
        "latency_avg_s": round(avg_latency, 3),
        "latency_min_s": round(min_latency, 3),
        "latency_max_s": round(max_latency, 3),
        "latency_p50_s": round(p50_latency, 3),
        "latency_p95_s": round(p95_latency, 3),
    }


def write_jsonl(results: list[EvaluationResult], output_path: Path):
    """Write results to JSONL file."""
    with open(output_path, "w") as f:
        for r in results:
            line = {
                "protocol_id": r.protocol_id,
                "response": r.response_json,
                "scores": {
                    "accuracy_at_1": r.accuracy_at_1,
                    "recall_at_3": r.recall_at_3,
                    "latency_s": round(r.latency_s, 3),
                    "ground_truth": r.ground_truth,
                    "top_prediction": r.top_prediction,
                    "top_3_predictions": r.top_3_predictions,
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_metrics_json(submission_name: str, metrics: dict, output_path: Path):
    """Write aggregated metrics to JSON file."""
    output_data = {
        "submission_name": submission_name,
        **metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def display_summary(
    results: list[EvaluationResult],
    metrics: dict,
    output_jsonl: Path,
    output_json: Path,
    console: Console,
):
    """Display a beautiful summary of the evaluation results."""
    if not results:
        console.print("[red]No results to display[/red]")
        return

    # Metrics table
    metrics_table = Table(
        title="[bold]Evaluation Metrics (small run)[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
    )
    metrics_table.add_column("Metric", style="cyan", width=20)
    metrics_table.add_column("Value", style="green", justify="right", width=15)

    metrics_table.add_row("Accuracy@1", f"{metrics['accuracy_at_1_percent']:.2f}%")
    metrics_table.add_row("Recall@3", f"{metrics['recall_at_3_percent']:.2f}%")
    metrics_table.add_row(
        "Total Protocols", f"[bold white]{metrics['total_protocols']}[/bold white]"
    )

    # Latency table
    latency_table = Table(
        title="[bold]Latency Statistics[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
    )
    latency_table.add_column("Statistic", style="cyan", width=20)
    latency_table.add_column("Value (s)", style="green", justify="right", width=15)

    latency_table.add_row("Average", f"{metrics['latency_avg_s']:.3f}")
    latency_table.add_row("Min", f"{metrics['latency_min_s']:.3f}")
    latency_table.add_row("Max", f"{metrics['latency_max_s']:.3f}")
    latency_table.add_row("P50 (Median)", f"{metrics['latency_p50_s']:.3f}")
    latency_table.add_row("P95", f"{metrics['latency_p95_s']:.3f}")

    console.print()
    console.print(metrics_table)
    console.print()
    console.print(latency_table)
    console.print()

    success_text = Text()
    success_text.append("✓ ", style="bold green")
    success_text.append("Results saved to:\n", style="white")
    success_text.append(f"  JSONL:   {output_jsonl}\n", style="bold cyan")
    success_text.append(f"  Metrics: {output_json}", style="bold cyan")
    console.print(Panel(success_text, border_style="green"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diagnostic accuracy on first 40 protocols (quick sanity check)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  python small_evaluate.py --endpoint http://localhost:8000/diagnose --dataset-dir ./data --name my_submission
  python small_evaluate.py -e http://api.example.com/diagnose -d ./protocols -n team_alpha -p 10
        """,
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="Submission/project name (used for output file naming)",
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        required=True,
        help="URL of the diagnostic endpoint",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        required=True,
        type=Path,
        help="Directory containing JSON protocol files",
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        default=2,
        help="Number of simultaneous requests (default: 2)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/evals"),
        help="Output directory for results (default: data/evals)",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=SMALL_EVAL_LIMIT,
        help=f"Number of protocols to evaluate (default: {SMALL_EVAL_LIMIT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging (including full tracebacks for errors)",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    console = Console()

    if not args.dataset_dir.exists():
        console.print(
            f"[red]Error: Dataset directory '{args.dataset_dir}' does not exist[/red]"
        )
        return 1

    if not args.dataset_dir.is_dir():
        console.print(f"[red]Error: '{args.dataset_dir}' is not a directory[/red]")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = asyncio.run(
        run_evaluation(
            endpoint=args.endpoint,
            dataset_dir=args.dataset_dir,
            parallelism=args.parallelism,
            limit=args.limit,
        )
    )

    if results:
        output_jsonl = args.output_dir / f"{args.name}.jsonl"
        output_json = args.output_dir / f"{args.name}_metrics.json"

        write_jsonl(results, output_jsonl)
        metrics = compute_metrics(results)
        write_metrics_json(args.name, metrics, output_json)
        display_summary(results, metrics, output_jsonl, output_json, console)

    return 0


if __name__ == "__main__":
    exit(main())
