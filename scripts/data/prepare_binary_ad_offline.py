#!/usr/bin/env python3
"""Prepare offline PVLA/SAM3 normal-reference assets for MPDD, DTD, and DAGM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glls import paths as glls_paths
from glls.eval.binary_ad import BINARY_AD_DATASETS
from glls.eval.binary_ad_offline import prepare_binary_ad_offline_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=[*BINARY_AD_DATASETS, "all"])
    parser.add_argument("--max_refs", type=int, default=4)
    parser.add_argument("--with_sam3", action="store_true", help="Load SAM3 and generate masked normal-reference cutouts.")
    parser.add_argument("--sam3_checkpoint", default=glls_paths.sam3_path())
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    datasets = BINARY_AD_DATASETS if args.dataset == "all" else (args.dataset,)
    roots = {dataset: Path(glls_paths.binary_ad_root(dataset)).expanduser() for dataset in datasets}
    sam_engine = None
    if args.with_sam3:
        from glls.seg.sam3_engine import Sam3Engine

        device = None if args.device == "auto" else args.device
        sam_engine = Sam3Engine(args.sam3_checkpoint, device=device)

    manifest = prepare_binary_ad_offline_assets(
        dataset_roots=roots,
        datasets=datasets,
        max_refs=args.max_refs,
        sam_engine=sam_engine,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
