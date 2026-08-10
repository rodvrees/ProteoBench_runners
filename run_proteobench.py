#!/usr/bin/env python3
"""ProteoBench pipeline runner.

Usage:
    python run_proteobench.py [options]

Examples:
    # Dry run — show all planned jobs and their commands without executing
    python run_proteobench.py --dry-run

    # List available tools and datasets
    python run_proteobench.py --list-tools
    python run_proteobench.py --list-datasets

    # Run only DIA-NN on HYE_Astral
    python run_proteobench.py --tool diann --dataset HYE_Astral

    # Run everything enabled in config
    python run_proteobench.py

    # Use a custom config file
    python run_proteobench.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

# Ensure runners package is importable when script is run directly
sys.path.insert(0, str(Path(__file__).parent))

from config_validator import validate_config
from runners import RUNNER_MAP
from runners.base import BaseRunner, RunResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

try:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    _RICH = True
except ImportError:
    _RICH = False


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_jobs(cfg: dict, tool_filter: str | None, dataset_filter: str | None) -> list[BaseRunner]:
    global_cfg = cfg.get("global") or {}
    search_params = cfg.get("search_params") or {}
    datasets = cfg.get("datasets") or {}
    tools = cfg.get("tools") or {}

    jobs: list[BaseRunner] = []

    for tool_name, tool_cfg in tools.items():
        if tool_filter and tool_name != tool_filter:
            continue

        RunnerClass = RUNNER_MAP.get(tool_name)
        if RunnerClass is None:
            logger.warning("No runner implemented for tool '%s'; skipping.", tool_name)
            continue

        for version_cfg in tool_cfg.get("versions", []):
            if not version_cfg.get("enabled", False):
                continue

            for dataset_name in tool_cfg.get("datasets", []):
                if dataset_filter and dataset_name != dataset_filter:
                    continue

                if dataset_name not in datasets:
                    logger.warning("Dataset '%s' listed under tool '%s' not found in datasets config; skipping.",
                                   dataset_name, tool_name)
                    continue

                dataset_cfg = datasets[dataset_name]
                runner = RunnerClass(
                    tool_cfg=tool_cfg,
                    dataset_name=dataset_name,
                    dataset_cfg=dataset_cfg,
                    version_cfg=version_cfg,
                    global_cfg=global_cfg,
                    search_params=search_params,
                )
                if not runner.is_compatible():
                    logger.debug("Skipping incompatible job: %s v%s on %s",
                                 tool_name, version_cfg["id"], dataset_name)
                    continue
                jobs.append(runner)

    return jobs


def preflight_all(jobs: list[BaseRunner]) -> bool:
    all_ok = True
    for job in jobs:
        errors = job.preflight_check()
        if errors:
            all_ok = False
            for err in errors:
                logger.error("[%s v%s / %s] preflight FAIL: %s",
                             job.tool_name, job.version_id, job.dataset_name, err)
    return all_ok


def write_summary(results: list[RunResult], output_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"run_summary_{ts}.tsv"
    fieldnames = ["tool", "version", "dataset", "success", "skipped", "runtime_s", "output_dir", "error_msg"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in results:
            writer.writerow({
                "tool": r.tool,
                "version": r.version,
                "dataset": r.dataset,
                "success": r.success,
                "skipped": r.skipped,
                "runtime_s": f"{r.runtime_s:.1f}",
                "output_dir": r.output_dir,
                "error_msg": r.error_msg,
            })
    return summary_path


def print_summary(results: list[RunResult]) -> None:
    W = 76
    print("\n" + "=" * W)
    print(f"  {'TOOL':<15} {'VERSION':<10} {'DATASET':<24} {'STATUS':<7} {'TIME':>6}")
    print("-" * W)
    for r in sorted(results, key=lambda x: (x.tool, x.version, x.dataset)):
        if r.skipped:
            status = "SKIP"
        elif r.success:
            status = "OK"
        else:
            status = "FAIL"
        time_str = f"{r.runtime_s:.0f}s"
        print(f"  {r.tool:<15} {r.version:<10} {r.dataset:<24} {status:<7} {time_str:>6}")
        if not r.success and not r.skipped and r.error_msg:
            print(f"    Error: {r.error_msg}")
        if not r.success and not r.skipped and r.stderr_log:
            print(f"    Log:   {r.stderr_log}")
    print("=" * W)
    n_ok   = sum(r.success and not r.skipped for r in results)
    n_skip = sum(r.skipped for r in results)
    n_fail = sum(not r.success for r in results)
    print(f"  Result: {n_ok} succeeded, {n_skip} skipped, {n_fail} failed")
    print("=" * W + "\n")


def list_tools(cfg: dict) -> None:
    tools = cfg.get("tools") or {}
    datasets = cfg.get("datasets") or {}
    print(f"\nAvailable tools (from config.yaml):\n")
    print(f"  {'TOOL':<14} {'VERSIONS':>9}  {'ENABLED':>8}  DATASETS")
    print("  " + "-" * 60)
    for tool_name, tool_cfg in tools.items():
        versions = tool_cfg.get("versions", [])
        n_total   = len(versions)
        n_enabled = sum(1 for v in versions if v.get("enabled", False))
        ds_list = tool_cfg.get("datasets") or []
        # Only count datasets that exist in the datasets section
        ds_valid = [d for d in ds_list if d in datasets]
        ds_str = ", ".join(ds_valid) if ds_valid else "(none)"
        print(f"  {tool_name:<14} {n_total:>4} total  {n_enabled:>4} enabled  {ds_str}")
    print()


def list_datasets(cfg: dict) -> None:
    datasets = cfg.get("datasets") or {}
    print(f"\nAvailable datasets (from config.yaml):\n")
    print(f"  {'NAME':<28} {'ACQ':<5} {'FMT':<6} {'INSTRUMENT':<12} PATH")
    print("  " + "-" * 80)
    for ds_name, ds in datasets.items():
        acq  = ds.get("acquisition", "?")
        fmt  = ds.get("format", "?")
        inst = ds.get("instrument", "?")
        path = ds.get("path", "?")
        print(f"  {ds_name:<28} {acq:<5} {fmt:<6} {inst:<12} {path}")
    print()


def show_dry_run_commands(jobs: list[BaseRunner]) -> None:
    print("\nDry run — commands that would be executed:\n")
    for job in jobs:
        print(f"  [{job.tool_name} v{job.version_id} / {job.dataset_name}]")
        try:
            input_files = job.get_input_files()
            fasta = Path(job.dataset_cfg["fasta"])
            output_dir = job.make_output_dir()
            # Mirror run(): extra_args are appended after build_command, so the
            # preview has to add them too or it shows a command that never runs.
            cmd = list(job.build_command(input_files, fasta, output_dir)) + job.extra_args()
            print("  " + shlex.join(str(c) for c in cmd))
        except Exception as exc:
            print(f"  (command preview unavailable: {exc})")
        print()


def _run_with_rich(jobs: list[BaseRunner], max_workers: int) -> list[RunResult]:
    results: list[RunResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        overall = progress.add_task("Running jobs", total=len(jobs))
        task_ids = {
            pool_future: progress.add_task(
                f"{job.tool_name} v{job.version_id} / {job.dataset_name}",
                total=1,
                start=False,
            )
            for pool_future, job in [
                (None, job) for job in jobs  # placeholder; will be replaced below
            ]
        }

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_job: dict = {}
            rich_task: dict = {}
            for job in jobs:
                f = pool.submit(job.run)
                tid = progress.add_task(
                    f"{job.tool_name} v{job.version_id} / {job.dataset_name}",
                    total=1,
                )
                progress.start_task(tid)
                future_to_job[f] = job
                rich_task[f] = tid

            for future in as_completed(future_to_job):
                result = future.result()
                results.append(result)
                tid = rich_task[future]
                progress.advance(tid)
                progress.advance(overall)
                status = "SKIP" if result.skipped else ("OK" if result.success else "FAIL")
                logger.info("[%s] %-15s v%-8s  %s  %.1fs",
                            status, result.tool, result.version, result.dataset, result.runtime_s)

    return results


def _run_plain(jobs: list[BaseRunner], max_workers: int) -> list[RunResult]:
    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(job.run): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            status = "SKIP" if result.skipped else ("OK" if result.success else "FAIL")
            logger.info("[%s] %-15s v%-8s  %s  %.1fs  → %s",
                        status, result.tool, result.version, result.dataset,
                        result.runtime_s, result.output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run proteomics search engines on ProteoBench datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_proteobench.py --list-tools          # show configured tools
  python run_proteobench.py --list-datasets       # show configured datasets
  python run_proteobench.py --dry-run             # preview commands without running
  python run_proteobench.py --tool diann          # run only DIA-NN
  python run_proteobench.py                       # run all enabled jobs
""",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to YAML config (default: config.yaml next to this script)")
    parser.add_argument("--tool", help="Run only this tool (e.g. diann, sage, alphadia)")
    parser.add_argument("--dataset", help="Run only this dataset (e.g. HYE_Astral)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show planned jobs and full CLI commands without executing anything")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip preflight checks and run all enabled jobs regardless")
    parser.add_argument("--list-tools", action="store_true",
                        help="List all configured tools with version and dataset counts, then exit")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all configured datasets with format and path, then exit")
    args = parser.parse_args()

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        if (Path(__file__).parent / "config.template.yaml").exists():
            logger.error(
                "Copy the template to get started: "
                "cp config.template.yaml config.yaml   then edit the paths inside."
            )
        sys.exit(1)

    cfg = load_config(args.config)

    # --list-tools and --list-datasets work even with a partially broken config
    if args.list_tools:
        list_tools(cfg)
        sys.exit(0)
    if args.list_datasets:
        list_datasets(cfg)
        sys.exit(0)

    validation_errors = validate_config(cfg, args.config)
    if validation_errors:
        print(f"\nConfiguration errors in {args.config}:\n")
        for i, err in enumerate(validation_errors, 1):
            print(f"  [{i}] {err}")
        print(f"\nEdit {args.config} and fix the errors above before running.\n")
        sys.exit(1)

    global_cfg = cfg.get("global") or {}
    output_dir = Path(global_cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = global_cfg.get("max_parallel_jobs", 2)

    jobs = build_jobs(cfg, tool_filter=args.tool, dataset_filter=args.dataset)
    if not jobs:
        logger.warning("No enabled jobs found. Check config.yaml enabled flags and filters.")
        logger.warning("Run: python run_proteobench.py --list-tools   to see which tools are enabled.")
        sys.exit(0)

    logger.info("Found %d job(s) to run.", len(jobs))
    for j in jobs:
        logger.info("  %-15s v%-8s  %s", j.tool_name, j.version_id, j.dataset_name)

    if not args.no_preflight:
        logger.info("Running preflight checks...")
        ok = preflight_all(jobs)
        if not ok:
            logger.error("Preflight checks failed. Fix the errors above or use --no-preflight to skip.")
            sys.exit(1)
        logger.info("All preflight checks passed.")

    if args.dry_run:
        show_dry_run_commands(jobs)
        logger.info("Dry run complete. No jobs were executed.")
        sys.exit(0)

    if _RICH:
        results = _run_with_rich(jobs, max_workers)
    else:
        results = _run_plain(jobs, max_workers)

    print_summary(results)
    summary_path = write_summary(results, output_dir)

    n_skip = sum(r.skipped for r in results)
    n_ok   = sum(r.success and not r.skipped for r in results)
    n_fail = sum(not r.success for r in results)
    logger.info("Done. %d ran OK, %d skipped, %d failed. Summary: %s",
                n_ok, n_skip, n_fail, summary_path)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
