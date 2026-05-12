"""Post-training inspection of a SC-GS-default run.

Produces:
  - metrics_history.json: per-eval-iteration PSNR/SSIM/LPIPS on train+test splits
  - metrics_curve.png:    PSNR-vs-iteration plot (test split)
  - renders/<iter>/{train,test}/video.mp4: SC-GS-rendered videos (test = held-out)
  - renders/<iter>_time/test/video.mp4:    dense time-interpolation (failure surface)
  - REPORT.md:            short markdown summary

Designed for the W1 "reproduce SC-GS failure on articulated motion" experiment.
The render.py invocation reads cfg_args from <run_dir>/cfg_args so only
--model_path + --mode + --iteration are passed.

Usage:
    conda activate scgs
    python scripts/inspect_scgs_failure.py \\
        --run_dir outputs/jumpingjacks_scgs_default_node \\
        --source_path data/dnerf/jumpingjacks \\
        --iteration 30000
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCGS_DIR = REPO_ROOT / "third_party" / "SC-GS"

EVAL_LINE_RE = re.compile(
    r"\[ITER (?P<it>\d+)\] Evaluating (?P<split>train|test): "
    r"L1 (?P<l1>[\d.]+) "
    r"PSNR (?P<psnr>[\d.]+) "
    r"SSIM (?P<ssim>[\d.]+) "
    r"LPIPS (?P<lpips>[\d.]+) "
    r"MS SSIM (?P<msssim>[\d.]+) "
    r"ALEX_LPIPS (?P<alex>[\d.]+)"
)


def parse_train_log(log_path: Path) -> dict:
    """Split tqdm-overwritten log on \\r, extract eval lines, dedupe by (iter, split)."""
    raw = log_path.read_text(errors="replace")
    lines = raw.replace("\r", "\n").splitlines()
    seen: dict[tuple[int, str], dict] = {}
    for line in lines:
        m = EVAL_LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        key = (int(d["it"]), d["split"])
        seen[key] = {
            "iter": int(d["it"]),
            "split": d["split"],
            "l1": float(d["l1"]),
            "psnr": float(d["psnr"]),
            "ssim": float(d["ssim"]),
            "lpips": float(d["lpips"]),
            "ms_ssim": float(d["msssim"]),
            "alex_lpips": float(d["alex"]),
        }
    rows = sorted(seen.values(), key=lambda r: (r["iter"], r["split"]))
    by_split = {"train": [], "test": []}
    for r in rows:
        by_split[r["split"]].append(r)
    final_test = by_split["test"][-1] if by_split["test"] else None
    best_test = max(by_split["test"], key=lambda r: r["psnr"]) if by_split["test"] else None
    return {
        "history": rows,
        "by_split": by_split,
        "final_test": final_test,
        "best_test": best_test,
        "num_evals_test": len(by_split["test"]),
    }


def plot_curve(metrics: dict, out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[inspect] matplotlib unavailable; skipping curve plot")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split, color in (("train", "C0"), ("test", "C1")):
        rows = metrics["by_split"][split]
        if not rows:
            continue
        it = [r["iter"] for r in rows]
        axes[0].plot(it, [r["psnr"] for r in rows], color, label=split, marker="o", ms=3)
        axes[1].plot(it, [r["ssim"] for r in rows], color, label=split, marker="o", ms=3)
        axes[2].plot(it, [r["lpips"] for r in rows], color, label=split, marker="o", ms=3)
    for ax, ylabel in zip(axes, ("PSNR ↑", "SSIM ↑", "LPIPS ↓")):
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("SC-GS-default — convergence curves")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def call_render(run_dir: Path, source_path: Path, iteration: int, mode: str,
                deform_type: str = "node") -> int:
    """Run SC-GS render.py from third_party/SC-GS with the trained model.

    render.py reads cfg_args from run_dir/cfg_args, so we only need to pass
    the minimal flags. --deform_type must be passed explicitly: render.py uses
    ModelParams(sentinel=True), and arguments/__init__.py:162's endswith check
    fires on cmdline args BEFORE the cfg_args merge -- so a sentinel None for
    deform_type crashes there. Captures stdout/stderr into render_<mode>.log.
    """
    log = run_dir / f"render_{mode}.log"
    # If model_path already has the deform_type suffix, strip it: render.py's
    # endswith-check re-appends `_{deform_type}` if missing, so passing the
    # un-suffixed path keeps cfg_args lookup pointing at the right dir.
    suffix = f"_{deform_type}"
    mp = run_dir.resolve()
    if str(mp).endswith(suffix):
        mp_arg = Path(str(mp)[: -len(suffix)])
    else:
        mp_arg = mp
    cmd = [
        "python",
        "-u",
        "render.py",
        "--source_path",
        str(source_path.resolve()),
        "--model_path",
        str(mp_arg),
        "--deform_type",
        deform_type,
        "--iteration",
        str(iteration),
        "--mode",
        mode,
        "--skip_train",
    ]
    with log.open("w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(SCGS_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode


def write_report(run_dir: Path, metrics: dict, render_status: dict, out_md: Path) -> None:
    ft = metrics["final_test"]
    bt = metrics["best_test"]
    history = metrics["by_split"]["test"]

    def fmt(r: dict | None) -> str:
        if r is None:
            return "_(no eval line in log)_"
        return f"iter={r['iter']}  PSNR={r['psnr']:.3f}  SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}"

    HANDOFF_TARGET = 43.0  # docs/HANDOFF.md §4.3
    gap = None if ft is None else (HANDOFF_TARGET - ft["psnr"])

    lines = [
        f"# SC-GS failure inspection — {run_dir.name}",
        "",
        "## 1. Quantitative summary (test split)",
        "",
        f"- **Final**: {fmt(ft)}",
        f"- **Best**:  {fmt(bt)}",
        f"- **HANDOFF §4.3 target**: PSNR ~{HANDOFF_TARGET} (env-validation gate)",
        f"- **Gap from target**: {gap:+.2f}" if gap is not None else "",
        f"- Number of test-split eval points: {metrics['num_evals_test']}",
        "",
        "## 2. Convergence",
        "",
        "See `metrics_curve.png`. Convergence trajectory (test PSNR):",
        "",
        "| iter | PSNR | SSIM | LPIPS |",
        "|-----:|-----:|-----:|------:|",
    ]
    for r in history:
        lines.append(
            f"| {r['iter']} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} |"
        )
    lines += [
        "",
        "## 3. Qualitative renders",
        "",
    ]
    for mode, rc in render_status.items():
        ok = "✓" if rc == 0 else f"✗ (exit {rc})"
        lines.append(f"- mode=`{mode}`: {ok}")
    lines += [
        "",
        "Test-view video: `test/ours_<iter>/renders/` + `video.mp4`",
        "Time-interpolation video (dense temporal): `test/ours_<iter>_time/renders/` + `video.mp4`",
        "",
        "## 4. Failure analysis — what to look for",
        "",
        "SC-GS's known weakness on D-NeRF `jumpingjacks` (the W1 gate scene):",
        "1. **Joint-region blur** — Gaussians near elbow/knee/shoulder joints get",
        "   over-smoothed because SC-GS's vanilla ARAP treats all edges as equally rigid.",
        "2. **Limb-end ghosting** — fast-moving extremities (hands, feet) trail",
        "   residual Gaussians, producing dim halos in the time-interpolated video.",
        "3. **PSNR ceiling near 40-43** — this is what HANDOFF §6.4 names as the",
        "   '4D-Fly stated failure mode' for articulated motion. Our W1 gate.",
        "",
        "Open the two videos and visually note: blur at joints, ghosting on hands/feet,",
        "any rendering 'pop' between adjacent test timesteps.",
        "",
        "## 5. Next steps",
        "",
        "- Run `articulation_only` ablation through our hook (HANDOFF §4.6) and diff.",
        "- Once SAM 2 part labels exist (HANDOFF §4.7), the full inter-part angular",
        "  consistency metric (`motionprior.metrics.articulation`) is the headline",
        "  number to compare on.",
        "- If the articulation_only ablation does not visibly improve test-video",
        "  quality at joints, pivot to Option B/C from the survey (HANDOFF §6.4 stop-gate).",
        "",
    ]
    out_md.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, type=Path,
                   help="SC-GS model_path (e.g. outputs/jumpingjacks_scgs_default_node)")
    p.add_argument("--source_path", required=True, type=Path,
                   help="Scene root (e.g. data/dnerf/jumpingjacks)")
    p.add_argument("--iteration", type=int, default=30000,
                   help="Iteration checkpoint to render from (default 30000)")
    p.add_argument("--skip_render", action="store_true",
                   help="Skip the slow SC-GS render.py calls (parsing/curves only)")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    insp = run_dir / "inspection"
    insp.mkdir(exist_ok=True)
    log_path = run_dir / "train.log"
    if not log_path.exists():
        print(f"[inspect] FATAL: {log_path} not found")
        return 2

    print(f"[inspect] parsing {log_path}...")
    metrics = parse_train_log(log_path)
    (insp / "metrics_history.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[inspect]   {metrics['num_evals_test']} test-split eval points")
    if metrics["final_test"]:
        ft = metrics["final_test"]
        print(f"[inspect]   final test PSNR = {ft['psnr']:.3f} at iter {ft['iter']}")

    print(f"[inspect] plotting convergence curves -> {insp / 'metrics_curve.png'}")
    plot_curve(metrics, insp / "metrics_curve.png")

    render_status: dict[str, int] = {}
    if not args.skip_render:
        for mode in ("render", "time"):
            print(f"[inspect] running SC-GS render.py --mode {mode} (may take a few min)...")
            rc = call_render(run_dir, args.source_path, args.iteration, mode)
            render_status[mode] = rc
            print(f"[inspect]   exit {rc}")
    else:
        print("[inspect] --skip_render set; not invoking render.py")

    print(f"[inspect] writing REPORT.md -> {insp / 'REPORT.md'}")
    write_report(run_dir, metrics, render_status, insp / "REPORT.md")

    print(f"[inspect] done. Artifacts: {insp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
