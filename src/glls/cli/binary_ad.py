from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from glls import paths as glls_paths
from glls.eval.binary_ad import (
    BINARY_AD_DATASETS,
    aggregate_binary_ad_summaries,
    discovery_summary,
    load_binary_ad_threshold_table,
    run_binary_ad_evaluation,
    run_binary_ad_pipeline,
)
from glls.eval.binary_ad_offline import load_binary_ad_offline_manifest, prepare_binary_ad_offline_assets


LOCALIZER_DATASET_NAMES = {
    "mpdd": "mpdd",
    "dtd": "DTD",
    "dagm": "DAGM_KaggleUpload",
}


def build_localizer(args, dataset: str | None = None):
    from glls.models.localizer import ABounD_Localizer, AdaptCLIP_Localizer

    dataset_key = dataset or args.dataset
    dataset_name = LOCALIZER_DATASET_NAMES.get(dataset_key, dataset_key)
    device = None if args.device == "auto" else args.device
    checkpoint_path = args.checkpoint_path
    if args.localizer == "adaptclip" and not checkpoint_path:
        checkpoint_path = str(Path(glls_paths.adaptclip_root()) / "checkpoints" / f"{args.adaptclip_checkpoint_domain}_epoch_15.pth")
    if args.localizer == "abound" and not checkpoint_path:
        checkpoint_path = glls_paths.abound_model_path()
    image_size = int(args.image_size or (518 if args.localizer == "adaptclip" else 336))
    localizer_args = SimpleNamespace(
        dataset=dataset_name,
        image_size=image_size,
        k_shot=args.k_shot,
        checkpoint_path=checkpoint_path,
        save_path=args.save_path,
    )
    if args.localizer == "adaptclip":
        return AdaptCLIP_Localizer(localizer_args, device=device, pretrained_model=args.pretrained_model)
    return ABounD_Localizer(localizer_args, device=device)


def build_sam3_engine(args):
    from glls.seg.sam3_engine import Sam3Engine

    device = None if args.device == "auto" else args.device
    return Sam3Engine(args.sam3_checkpoint, device=device)


def build_final_verifier(args):
    if args.final_verifier != "qwen3":
        return None

    from glls.cli.run import UnifiedVLMInference
    from glls.eval.binary_ad_qwen3 import BinaryADQwen3Verifier, BinaryADQwen3VerifierConfig

    vlm_engine = UnifiedVLMInference(
        model_path=args.model_path,
        device=_vlm_device(args),
        model_type=args.model_type,
        use_vllm=args.use_vllm,
        max_tokens=args.vlm_context_tokens,
    )
    verifier = BinaryADQwen3Verifier(
        vlm_engine,
        config=BinaryADQwen3VerifierConfig(
            max_crops=args.qwen3_max_crops,
            max_refs=args.qwen3_max_refs,
            max_tokens=args.qwen3_max_tokens,
            include_score_hint=bool(args.qwen3_score_hint and not args.qwen3_no_score_hint),
            fallback_to_threshold=not args.qwen3_no_fallback_threshold,
        ),
        model_name="qwen3",
    )
    return verifier.verify


def _vlm_device(args) -> str:
    if args.vlm_device:
        return args.vlm_device
    if args.device != "auto":
        return args.device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary anomaly-detection accuracy for MPDD, DTD, and DAGM.")
    parser.add_argument("--dataset", required=True, choices=["mpdd", "dtd", "dagm", "all"])
    parser.add_argument("--dataset_root", default="", help="Dataset root. Defaults to the matching GLLS_* root.")
    parser.add_argument("--category", default="", help="Optional category/class filter.")
    parser.add_argument("--output_dir", default="", help="Output directory for evaluation files.")
    parser.add_argument(
        "--localizer",
        default="adaptclip",
        choices=["abound", "adaptclip"],
        help="Published default is AdaptCLIP. ABounD remains an explicit local-only option.",
    )
    parser.add_argument("--device", default="auto", help="Device string such as cuda:0 or cpu. Default lets the localizer choose.")
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=0, help="Default: 518 for AdaptCLIP, 336 for local ABounD.")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--adaptclip_checkpoint_domain", default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--save_path", default="", help="Local ABounD-only save path; unused by AdaptCLIP.")
    parser.add_argument("--pretrained_model", default="ViT-L/14@336px")
    parser.add_argument("--sam3_checkpoint", default=glls_paths.sam3_path())
    parser.add_argument("--normal_quantile", type=float, default=0.995)
    parser.add_argument("--mad_scale", type=float, default=6.0)
    parser.add_argument("--score_tail_fraction", type=float, default=0.01)
    parser.add_argument(
        "--binary_score_source",
        default="auto",
        choices=["auto", "localizer_image", "heatmap_tail", "max_image_heatmap", "mean_image_heatmap"],
    )
    parser.add_argument("--threshold_policy", default="normal_robust", choices=["normal_robust", "normal_quantile", "normal_median_mad", "fixed", "table"])
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_table", default="", help="JSON file with per-dataset/category thresholds.")
    parser.add_argument("--calibration_shots", type=int, default=-1, help="-1 uses all train-normal scores; 0 uses no train-normal calibration and requires --threshold_policy fixed; positive values use that many normal scores per category.")
    parser.add_argument("--max_region_proposals", type=int, default=3)
    parser.add_argument("--max_pvla_refs", type=int, default=3)
    parser.add_argument("--skip_offline_assets", action="store_true", help="Do not prepare/load offline PVLA/SAM3 normal-reference assets before evaluation.")
    parser.add_argument("--reuse_offline_assets", action="store_true", help="Load an existing binary AD offline manifest instead of rebuilding PVLA/SAM3 assets.")
    parser.add_argument("--no_sam3", action="store_true", help="Do not load SAM3 during online evaluation/offline asset preparation.")
    parser.add_argument("--no_save_evidence_images", action="store_true")
    parser.add_argument("--final_verifier", default="threshold", choices=["threshold", "qwen3"])
    parser.add_argument("--final_verifier_policy", default="anomaly_or", choices=["replace", "anomaly_or", "audit"])
    parser.add_argument("--model_path", default=glls_paths.vlm_model_path())
    parser.add_argument("--model_type", default="qwen3")
    parser.add_argument("--vlm_device", default="", help="Device for the final VLM verifier. Defaults to --device, or cuda:0 when available.")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vlm_context_tokens", type=int, default=8192)
    parser.add_argument("--qwen3_max_tokens", type=int, default=512)
    parser.add_argument("--qwen3_max_crops", type=int, default=2)
    parser.add_argument("--qwen3_max_refs", type=int, default=4)
    parser.add_argument("--qwen3_score_hint", action="store_true", help="Include the localizer score/threshold decision in the Qwen3 prompt. Disabled by default.")
    parser.add_argument("--qwen3_no_score_hint", action="store_true", help="Compatibility switch; score hints are already disabled unless --qwen3_score_hint is set.")
    parser.add_argument("--qwen3_no_fallback_threshold", action="store_true")
    parser.add_argument("--max_test_samples_per_category", type=int, default=0, help="Debug/smoke limit. 0 evaluates every test image.")
    parser.add_argument("--resume", action="store_true", help="Skip image paths already present in predictions.jsonl and append new rows as they finish.")
    parser.add_argument(
        "--discovery_only",
        action="store_true",
        help="Only parse dataset structure and labels; do not load the localizer.",
    )
    args = parser.parse_args()

    if args.final_verifier == "qwen3" and args.no_save_evidence_images:
        parser.error("--final_verifier qwen3 requires evidence images; remove --no_save_evidence_images")

    if args.dataset == "all":
        if args.category:
            parser.error("--category is only supported for a single dataset run")
        if args.dataset_root:
            parser.error("--dataset_root is only supported for a single dataset run")
        roots = {dataset: Path(glls_paths.binary_ad_root(dataset)).expanduser() for dataset in BINARY_AD_DATASETS}
        if args.discovery_only:
            summaries = [discovery_summary(dataset=dataset, root=root) for dataset, root in roots.items()]
            print(json.dumps({
                "pipeline": "mpdd_dtd_dagm_binary_ad_discovery",
                "datasets": summaries,
                "aggregate": aggregate_binary_ad_summaries(summaries),
            }, ensure_ascii=False, indent=2))
            return
        output_dir = Path(args.output_dir or "outputs/binary_ad/mpdd_dtd_dagm").expanduser()
        discoveries = [discovery_summary(dataset=dataset, root=root) for dataset, root in roots.items()]
        missing = [row for row in discoveries if not row["root_exists"] or not row["train_normal_total"] or not row["test_total"]]
        if missing:
            parser.error(
                "one or more datasets are not ready for evaluation: "
                + ", ".join(f"{row['dataset']}@{row['root']}" for row in missing)
            )
        shared_sam_engine = None if args.no_sam3 else build_sam3_engine(args)
        offline_assets = None
        if not args.skip_offline_assets:
            offline_assets = (
                load_binary_ad_offline_manifest()
                if args.reuse_offline_assets
                else prepare_binary_ad_offline_assets(
                    dataset_roots=roots,
                    sam_engine=shared_sam_engine,
                    max_refs=args.max_pvla_refs,
                )
            )
        threshold_table = load_binary_ad_threshold_table(args.threshold_table)
        final_verifier = build_final_verifier(args)
        summary = run_binary_ad_pipeline(
            dataset_roots=roots,
            output_dir=output_dir,
            localizer_factory=lambda dataset: build_localizer(args, dataset=dataset),
            sam_engine_factory=lambda _dataset: shared_sam_engine,
            normal_quantile=args.normal_quantile,
            mad_scale=args.mad_scale,
            score_tail_fraction=args.score_tail_fraction,
            binary_score_source=args.binary_score_source,
            threshold_policy=args.threshold_policy,
            fixed_threshold=args.fixed_threshold,
            threshold_table=threshold_table,
            localizer_name=args.localizer,
            threshold_shot=args.k_shot,
            calibration_shots=args.calibration_shots,
            max_region_proposals=args.max_region_proposals,
            max_pvla_refs=args.max_pvla_refs,
            save_evidence_images=not args.no_save_evidence_images,
            offline_assets=offline_assets,
            final_verifier_factory=(lambda _dataset: final_verifier) if final_verifier else None,
            final_verifier_policy=args.final_verifier_policy,
            max_test_samples_per_category=args.max_test_samples_per_category,
            resume=args.resume,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    root = Path(args.dataset_root or glls_paths.binary_ad_root(args.dataset)).expanduser()
    if args.discovery_only:
        summary = discovery_summary(dataset=args.dataset, root=root, category=args.category or None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not args.output_dir:
        parser.error("--output_dir is required unless --discovery_only is set")

    discovered = discovery_summary(dataset=args.dataset, root=root, category=args.category or None)
    if not discovered["root_exists"]:
        parser.error(f"dataset root does not exist: {root}")
    if not discovered["train_normal_total"] or not discovered["test_total"]:
        parser.error(
            "dataset discovery found no usable train/test split; "
            "run with --discovery_only to inspect the parsed layout"
        )

    sam_engine = None if args.no_sam3 else build_sam3_engine(args)
    offline_assets = None
    if not args.skip_offline_assets:
        offline_assets = (
            load_binary_ad_offline_manifest()
            if args.reuse_offline_assets
            else prepare_binary_ad_offline_assets(
                dataset_roots={args.dataset: root},
                datasets=[args.dataset],
                sam_engine=sam_engine,
                max_refs=args.max_pvla_refs,
            )
        ) or load_binary_ad_offline_manifest()

    localizer = build_localizer(args)
    threshold_table = load_binary_ad_threshold_table(args.threshold_table)
    final_verifier = build_final_verifier(args)
    summary = run_binary_ad_evaluation(
        dataset=args.dataset,
        root=root,
        localizer=localizer,
        output_dir=Path(args.output_dir).expanduser(),
        category=args.category or None,
        normal_quantile=args.normal_quantile,
        mad_scale=args.mad_scale,
        score_tail_fraction=args.score_tail_fraction,
        binary_score_source=args.binary_score_source,
        threshold_policy=args.threshold_policy,
        fixed_threshold=args.fixed_threshold,
        threshold_table=threshold_table,
        localizer_name=args.localizer,
        threshold_shot=args.k_shot,
        calibration_shots=args.calibration_shots,
        sam_engine=sam_engine,
        max_region_proposals=args.max_region_proposals,
        max_pvla_refs=args.max_pvla_refs,
        save_evidence_images=not args.no_save_evidence_images,
        offline_assets=offline_assets,
        final_verifier=final_verifier,
        final_verifier_policy=args.final_verifier_policy,
        max_test_samples_per_category=args.max_test_samples_per_category,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
