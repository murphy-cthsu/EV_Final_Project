"""Aggregate eval.json files into a Markdown ablation table + CSV.

Reads every ``<runs_root>/*/eval.json`` and groups by (scene, ablation),
emitting:
  * Markdown table — for paste into the paper / weekly progress note
  * CSV — for spreadsheet / downstream analysis
  * Summary of missing runs

Usage:
    python scripts/aggregate_results.py \\
        --runs_root outputs/ \\
        --out_md results.md \\
        --out_csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ABLATIONS_ORDER = [
    "scgs_default",
    "articulation_only",
    "gating_only",
    "curriculum_only",
    "gating_curriculum",
    "ours_minus_articulation",
    "full",
]

METRICS_ORDER = [
    "psnr",
    "ssim",
    "lpips",
    "inter_part_angular_consistency_mean",
    "num_articulated_parts",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="outputs", help="Directory containing per-run subdirs.")
    p.add_argument("--out_md", default="results.md", help="Markdown table output path.")
    p.add_argument("--out_csv", default="results.csv", help="CSV output path.")
    p.add_argument(
        "--metric_for_table", default="inter_part_angular_consistency_mean",
        choices=METRICS_ORDER,
        help="Metric to highlight in the per-scene table.",
    )
    return p.parse_args()


def load_eval(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": str(exc)}


def extract_metrics(eval_json: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    img = eval_json.get("image_metrics", {})
    for k in ("psnr", "ssim", "lpips"):
        v = img.get(k, float("nan"))
        out[k] = float(v) if isinstance(v, (int, float)) else float("nan")

    art = eval_json.get("articulation_metrics", {})
    v = art.get("inter_part_angular_consistency_mean", float("nan"))
    out["inter_part_angular_consistency_mean"] = (
        float(v) if isinstance(v, (int, float)) else float("nan")
    )

    sim = eval_json.get("sim_bridge", {})
    out["num_articulated_parts"] = float(sim.get("num_articulated_parts", 0))
    return out


def parse_run_name(run_dir_name: str) -> tuple[str, str] | None:
    """Convention: ``<scene>_<ablation>``.

    Ablation names may contain underscores (``gating_curriculum``), so we
    match against the known ABLATIONS_ORDER suffix list.
    """
    for ab in sorted(ABLATIONS_ORDER, key=len, reverse=True):
        suffix = f"_{ab}"
        if run_dir_name.endswith(suffix):
            scene = run_dir_name[: -len(suffix)]
            return scene, ab
    return None


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"[aggregate] {runs_root} does not exist; nothing to aggregate.")
        return 0

    results: dict[tuple[str, str], dict[str, float]] = {}
    missing: list[tuple[str, str]] = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        scene, ablation = parsed
        eval_path = run_dir / "eval.json"
        if not eval_path.exists():
            missing.append((scene, ablation))
            continue
        results[(scene, ablation)] = extract_metrics(load_eval(eval_path))

    # Discover scenes + ablations actually present
    scenes = sorted({s for (s, _) in results.keys()})
    ablations_present = [a for a in ABLATIONS_ORDER if any(a == k[1] for k in results)]

    # ----- Markdown table -----
    lines = [
        f"# Ablation results ({args.metric_for_table})",
        "",
        f"Aggregated from {len(results)} eval.json files under `{runs_root}`.",
        "",
    ]
    if missing:
        lines.append(f"**Missing runs** (no eval.json found): {len(missing)}")
        for s, a in sorted(missing):
            lines.append(f"  - {s} / {a}")
        lines.append("")

    if scenes and ablations_present:
        # Wide-form table: rows = scenes, cols = ablations
        header = "| scene | " + " | ".join(ablations_present) + " |"
        sep = "|---|" + "|".join(["---"] * len(ablations_present)) + "|"
        lines.append(header)
        lines.append(sep)
        for scene in scenes:
            row = [scene]
            for ab in ablations_present:
                v = results.get((scene, ab), {}).get(args.metric_for_table, float("nan"))
                row.append(f"{v:.4f}" if not math.isnan(v) else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Per-scene multi-metric blocks
        lines.append("## Full per-scene metrics")
        lines.append("")
        for scene in scenes:
            lines.append(f"### {scene}")
            lines.append("")
            mhdr = "| ablation | " + " | ".join(METRICS_ORDER) + " |"
            lines.append(mhdr)
            lines.append("|---|" + "|".join(["---"] * len(METRICS_ORDER)) + "|")
            for ab in ablations_present:
                row = [ab]
                metrics = results.get((scene, ab), {})
                for m in METRICS_ORDER:
                    v = metrics.get(m, float("nan"))
                    if math.isnan(v):
                        row.append("-")
                    elif m == "num_articulated_parts":
                        row.append(f"{int(v)}")
                    else:
                        row.append(f"{v:.4f}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    md = "\n".join(lines)
    Path(args.out_md).write_text(md)
    print(f"[aggregate] wrote {args.out_md}")

    # ----- CSV -----
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "ablation"] + METRICS_ORDER)
        for (scene, ab), metrics in sorted(results.items()):
            row = [scene, ab]
            for m in METRICS_ORDER:
                v = metrics.get(m, float("nan"))
                row.append("" if math.isnan(v) else f"{v}")
            w.writerow(row)
    print(f"[aggregate] wrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
