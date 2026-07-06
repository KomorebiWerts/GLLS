#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from glls import paths as glls_paths
from glls.eval.binary_ad import BINARY_AD_DATASETS, aggregate_binary_ad_summaries, discovery_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-comparable MPDD/DTD/DAGM binary AD with Qwen3 by category.")
    parser.add_argument("--shot", type=int, required=True, choices=[0, 1])
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--threshold_table", default="")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids.")
    parser.add_argument("--model_path", default=glls_paths.vlm_model_path())
    parser.add_argument("--model_type", default="qwen3")
    parser.add_argument("--max_refs", type=int, default=4)
    parser.add_argument("--max_crops", type=int, default=3)
    parser.add_argument("--final_verifier_policy", default="anomaly_or", choices=["replace", "anomaly_or", "audit"])
    parser.add_argument("--max_test_samples_per_category", type=int, default=0)
    parser.add_argument("--datasets", default=",".join(BINARY_AD_DATASETS))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    threshold_table = Path(args.threshold_table).expanduser() if args.threshold_table else None
    if threshold_table is not None and not threshold_table.is_file():
        raise SystemExit(f"Missing threshold table: {threshold_table}")
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_dir():
        raise SystemExit(f"Missing Qwen3 model directory: {model_path}")

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")

    jobs = _build_jobs(args=args, output_root=output_root, threshold_table=threshold_table, model_path=model_path)
    _write_json(output_root / "sweep_jobs.json", {"jobs": jobs, "gpus": gpus})
    if args.dry_run:
        print(json.dumps({"job_count": len(jobs), "gpus": gpus, "output_root": str(output_root)}, indent=2))
        return

    failures = _run_jobs(jobs, gpus=gpus, output_root=output_root)
    _aggregate_outputs(output_root=output_root, jobs=jobs, failures=failures)
    if failures:
        raise SystemExit(f"{len(failures)} category job(s) failed; see {output_root / 'sweep_failures.json'}")


def _build_jobs(
    *,
    args: argparse.Namespace,
    output_root: Path,
    threshold_table: Path | None,
    model_path: Path,
) -> list[dict[str, Any]]:
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    jobs = []
    for dataset in datasets:
        if dataset not in BINARY_AD_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        root = Path(glls_paths.binary_ad_root(dataset)).expanduser()
        summary = discovery_summary(dataset=dataset, root=root)
        for category_row in summary["categories"]:
            category = str(category_row["category"])
            out_dir = output_root / dataset / _safe_name(category)
            cmd = [
                sys.executable,
                "-m",
                "glls.cli.binary_ad",
                "--dataset",
                dataset,
                "--category",
                category,
                "--output_dir",
                str(out_dir),
                "--localizer",
                "adaptclip",
                "--k_shot",
                str(args.shot),
                "--device",
                "cuda:0",
                "--threshold_policy",
                "table",
                "--calibration_shots",
                str(args.shot),
                "--binary_score_source",
                "localizer_image",
                "--reuse_offline_assets",
                "--final_verifier",
                "qwen3",
                "--final_verifier_policy",
                args.final_verifier_policy,
                "--model_path",
                str(model_path),
                "--model_type",
                args.model_type,
                "--vlm_device",
                "cuda:0",
                "--qwen3_max_refs",
                str(args.max_refs),
                "--qwen3_max_crops",
                str(args.max_crops),
                "--resume",
            ]
            if threshold_table is not None:
                cmd.extend(["--threshold_table", str(threshold_table)])
            if args.max_test_samples_per_category > 0:
                cmd.extend(["--max_test_samples_per_category", str(args.max_test_samples_per_category)])
            jobs.append({
                "dataset": dataset,
                "category": category,
                "output_dir": str(out_dir),
                "test_total": int(category_row.get("test_total", 0) or 0),
                "cmd": cmd,
            })
    return jobs


def _run_jobs(jobs: list[dict[str, Any]], *, gpus: list[str], output_root: Path) -> list[dict[str, Any]]:
    pending = list(jobs)
    running: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed = 0
    start_time = time.time()

    while pending or running:
        while pending and len(running) < len(gpus):
            gpu = _free_gpu(gpus, running)
            if gpu is None:
                break
            job = pending.pop(0)
            log_path = Path(job["output_dir"]) / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_file = log_path.open("a", encoding="utf-8")
            log_file.write(f"\n=== start dataset={job['dataset']} category={job['category']} gpu={gpu} ===\n")
            log_file.flush()
            process = subprocess.Popen(
                job["cmd"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path.cwd()),
                text=True,
            )
            running.append({"process": process, "job": job, "gpu": gpu, "log_file": log_file, "start": time.time()})
            print(f"[start] gpu={gpu} {job['dataset']}/{job['category']} total={job['test_total']}", flush=True)

        time.sleep(5)
        still_running = []
        for item in running:
            process = item["process"]
            returncode = process.poll()
            if returncode is None:
                still_running.append(item)
                continue
            item["log_file"].write(f"=== end returncode={returncode} elapsed={time.time() - item['start']:.1f}s ===\n")
            item["log_file"].close()
            completed += 1
            job = item["job"]
            if returncode != 0:
                failure = {
                    "dataset": job["dataset"],
                    "category": job["category"],
                    "output_dir": job["output_dir"],
                    "returncode": returncode,
                    "gpu": item["gpu"],
                }
                failures.append(failure)
                print(f"[fail] {job['dataset']}/{job['category']} rc={returncode}", flush=True)
            else:
                elapsed = time.time() - item["start"]
                print(f"[done] {job['dataset']}/{job['category']} elapsed={elapsed:.1f}s ({completed}/{len(jobs)})", flush=True)
        running = still_running

    _write_json(output_root / "sweep_failures.json", failures)
    print(f"[sweep] completed={completed} failures={len(failures)} elapsed={time.time() - start_time:.1f}s", flush=True)
    return failures


def _free_gpu(gpus: list[str], running: list[dict[str, Any]]) -> str | None:
    used = {str(item["gpu"]) for item in running}
    for gpu in gpus:
        if gpu not in used:
            return gpu
    return None


def _aggregate_outputs(*, output_root: Path, jobs: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    dataset_rows = []
    for dataset in sorted({job["dataset"] for job in jobs}):
        dataset_jobs = [job for job in jobs if job["dataset"] == dataset]
        category_summaries = []
        prediction_rows = []
        for job in dataset_jobs:
            summary_path = Path(job["output_dir"]) / "summary.json"
            predictions_path = Path(job["output_dir"]) / "predictions.jsonl"
            if not summary_path.exists() or not predictions_path.exists():
                continue
            summary = _load_json(summary_path)
            category_summaries.extend(summary.get("categories") or [])
            prediction_rows.extend(_load_jsonl(predictions_path))

        total = len(prediction_rows)
        correct = sum(1 for row in prediction_rows if bool(row.get("correct")))
        dataset_out = output_root / "_combined" / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        discovered = discovery_summary(dataset=dataset, root=Path(glls_paths.binary_ad_root(dataset)).expanduser())
        dataset_summary = {
            "dataset": dataset,
            "root": glls_paths.binary_ad_root(dataset),
            "output_dir": str(dataset_out),
            "category_filter": "",
            "dataset_layout": discovered.get("dataset_layout", ""),
            "train_normal_total": int(discovered.get("train_normal_total", 0) or 0),
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "categories": category_summaries,
            "sam3_enabled": True,
            "pvla_enabled": True,
            "offline_assets_prepared": True,
            "final_verifier_enabled": True,
        }
        _write_json(dataset_out / "summary.json", dataset_summary)
        _write_jsonl(dataset_out / "predictions.jsonl", prediction_rows)
        dataset_rows.append(dataset_summary)

    pipeline_summary = aggregate_binary_ad_summaries(
        dataset_rows,
        output_dir=output_root / "_combined",
        final_verifier_enabled=True,
        write_files=True,
    )
    pipeline_summary["failed_category_jobs"] = failures
    _write_json(output_root / "pipeline_summary.json", pipeline_summary)
    _write_json(output_root / "dataset_summaries.json", dataset_rows)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._") or "item"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
