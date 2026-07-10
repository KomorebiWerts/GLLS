from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from glls import paths as glls_paths
from glls.weld.assets import build_weld_assets
from glls.weld.data import prepare_weld_dataset
from glls.weld.eval import evaluate_weld_adaptclip


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and evaluate the gear-weld dataset with GLLS, SAM3, and tiled AdaptCLIP."
    )
    parser.add_argument("--stage", choices=["all", "prepare", "assets", "evaluate"], default="all")
    parser.add_argument("--source_root", default="焊缝缺陷检测数据")
    parser.add_argument("--prepared_root", default="焊缝缺陷检测数据/prepared_glls")
    parser.add_argument("--output_dir", default="outputs/weld_ad")
    parser.add_argument("--shots", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sam_device",
        default="",
        help="SAM3 device. Defaults to the value of --device.",
    )
    parser.add_argument("--sam3_checkpoint", default=glls_paths.sam3_path())
    parser.add_argument("--skip_sam3_assets", action="store_true")
    parser.add_argument("--online_sam_limit", type=int, default=12)
    parser.add_argument("--checkpoint_domain", choices=["mvtec", "visa"], default="mvtec")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--pretrained_model", default="ViT-L/14@336px")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--tile_size", type=int, default=1600)
    parser.add_argument("--tile_overlap", type=float, default=0.25)
    parser.add_argument("--heatmap_max_side", type=int, default=768)
    parser.add_argument("--prompt_profile", choices=["generic", "weld", "spatter"], default="weld")
    parser.add_argument("--support_id", default="", help="1-shot support image id from manifest support_pool_ids.")
    parser.add_argument("--no_full_image", action="store_true", help="Use tiled local evidence only.")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--no_pvla_prior", action="store_true")
    parser.add_argument("--pvla_graph_path", default="")
    parser.add_argument("--target_recall", type=float, default=0.98)
    parser.add_argument(
        "--threshold_safety_margin",
        type=float,
        default=0.02,
        help="Relative downward threshold margin for the small validation anomaly set.",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Smoke-test limit per validation/test split.")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    sam_device = args.sam_device or args.device

    source_root = Path(args.source_root).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    manifest_path = prepared_root / "manifest.json"
    run_summary: dict[str, object] = {"stage": args.stage}

    if args.stage in {"all", "prepare"}:
        prepared = prepare_weld_dataset(source_root, prepared_root, seed=args.seed)
        manifest_path = Path(prepared["manifest_path"])
        run_summary["prepare"] = {
            "manifest_path": str(manifest_path),
            "counts": prepared["counts"],
            "review_sets": prepared.get("review_sets"),
            "data_policy": prepared.get("data_policy"),
        }
    elif not manifest_path.exists():
        parser.error(f"prepared manifest does not exist: {manifest_path}; run --stage prepare first")

    if args.stage in {"all", "assets"} and not args.skip_sam3_assets:
        assets = build_weld_assets(
            manifest_path,
            sam_checkpoint=args.sam3_checkpoint,
            device=sam_device,
        )
        run_summary["assets"] = {
            "graph_path": assets["graph_path"],
            "assets_manifest_path": assets["assets_manifest_path"],
            "reference_count": len(assets["references"]),
        }

    if args.stage in {"all", "evaluate"}:
        sam_engine = None
        if args.online_sam_limit > 0:
            from glls.seg.sam3_engine import Sam3Engine

            sam_engine = Sam3Engine(args.sam3_checkpoint, device=sam_device)
        evaluations = []
        for shot in args.shots:
            localizer = _build_adaptclip(args, shot=shot)
            shot_output = (
                output_root
                / args.checkpoint_domain
                / (
                    args.experiment_name
                    or f"{shot}shot_{args.prompt_profile}_tile{args.tile_size}_overlap{int(round(args.tile_overlap * 100)):02d}"
                )
            )
            result = evaluate_weld_adaptclip(
                manifest_path,
                localizer=localizer,
                output_dir=shot_output,
                shot=shot,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                heatmap_max_side=args.heatmap_max_side,
                target_recall=args.target_recall,
                threshold_safety_margin=args.threshold_safety_margin,
                resume=not args.no_resume,
                max_samples=args.max_samples,
                sam_engine=sam_engine,
                online_sam_limit=args.online_sam_limit,
                support_id=args.support_id,
                prompt_profile=args.prompt_profile,
                include_full_image=not args.no_full_image,
                experiment_name=args.experiment_name,
                use_pvla_prior=not args.no_pvla_prior,
                pvla_graph_path=Path(args.pvla_graph_path).expanduser() if args.pvla_graph_path else None,
            )
            evaluations.append(
                {
                    "shot": shot,
                    "output_dir": str(shot_output),
                    "data_status": result.get("data_status", "legacy labeled evaluation"),
                    "threshold_selection": result.get("threshold_selection"),
                    "image_metrics": result.get("image_metrics"),
                    "pixel_metrics": result.get("pixel_metrics"),
                    "review_count": len(result.get("review_rows") or []),
                }
            )
            del localizer
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        run_summary["evaluations"] = evaluations

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**run_summary, "run_summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


def _build_adaptclip(args: argparse.Namespace, *, shot: int):
    from glls.models.localizer import AdaptCLIP_Localizer

    checkpoint = args.checkpoint_path or str(
        Path(glls_paths.adaptclip_root()) / "checkpoints" / f"{args.checkpoint_domain}_epoch_15.pth"
    )
    localizer_args = SimpleNamespace(
        dataset="weld",
        image_size=args.image_size,
        k_shot=int(shot),
        checkpoint_path=checkpoint,
        save_path="",
    )
    return AdaptCLIP_Localizer(
        localizer_args,
        device=args.device,
        pretrained_model=args.pretrained_model,
    )


if __name__ == "__main__":
    main()
