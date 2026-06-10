"""D5 + D6: characterize SV4D inconsistency vs viewing angle from input view (elev0/az0).

D5 spatial fidelity : PSNR(SV4D, d-3dgs clean) baseplate-excluded, by elevation & azimuth.
D6 pose drift       : SV4D-digger vs canonical-digger silhouette IoU + centroid offset
                      (BOTH digger-only, no baseplate -> clean). canonical = our frozen
                      Gaussians rendered at each view; body dominates centroid so this
                      mostly captures view-synthesis pose drift.
Pure VGM characterization. Outputs curves + summary.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SCENE = REPO / "data/custom/lego_v3"
D3 = REPO / "outputs/custom/lego_v3_d3dgs_ref/renders"
CANON = REPO / "outputs/custom/lego_v3_canon_static_render"
OUT_FIG = REPO / "meetTW_checkpoint_0601/figs"

def load(p):
    im = np.asarray(Image.open(p), dtype=np.float32) / 255.0
    if im.shape[-1] == 4:
        a = im[..., 3]; rgb = im[..., :3]*a[..., None] + (1-a[..., None])
    else:
        rgb = im[..., :3]; a = (rgb < 0.97).any(-1).astype(np.float32)
    return rgb, a

def centroid(m):
    ys, xs = np.where(m)
    return None if len(xs)==0 else np.array([xs.mean(), ys.mean()])

def main():
    meta = json.loads((SCENE/"transforms_train.json").read_text())["frames"] + \
           json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    T = 21
    v2elev = {f["view_idx"]: f["elevation_deg"] for f in meta}
    v2az = {f["view_idx"]: f["azimuth_deg"] for f in meta}

    by_elev = defaultdict(lambda: defaultdict(list))
    by_az = defaultdict(lambda: defaultdict(list))
    n=0
    for f in meta:
        v, t = f["view_idx"], f["frame_idx"]; flat = v*T+t
        sv_p = None
        for sp in ("train","test"):
            c = SCENE/sp/f"r_{flat:05d}.png"
            if c.exists(): sv_p=c; break
        d3_p, cn_p = D3/f"{flat:05d}.png", CANON/f"{flat:05d}.png"
        if sv_p is None or not d3_p.exists() or not cn_p.exists(): continue
        sv_rgb, sv_a = load(sv_p); d3_rgb, d3_a = load(d3_p); cn_rgb, cn_a = load(cn_p)
        sv_fg, d3_fg, cn_fg = sv_a>0.5, d3_a>0.5, cn_a>0.5

        # D5: PSNR baseplate-excluded (baseplate = d3 fg not in sv fg)
        keep = ~(d3_fg & ~sv_fg)
        mse = (((sv_rgb-d3_rgb)**2).mean(-1)*keep).sum()/max(keep.sum(),1)
        p = -10*np.log10(max(mse,1e-12))

        # D6: SV4D digger vs canonical digger (both no baseplate)
        inter = (sv_fg & cn_fg).sum(); union = (sv_fg | cn_fg).sum()
        iou = inter/max(union,1)
        c_sv, c_cn = centroid(sv_fg), centroid(cn_fg)
        off = float(np.linalg.norm(c_sv-c_cn)) if (c_sv is not None and c_cn is not None) else np.nan

        e, az = v2elev[v], v2az[v]
        for store,key in ((by_elev,e),(by_az,az)):
            store[key]["psnr"].append(p); store[key]["iou"].append(iou)
            if not np.isnan(off): store[key]["off"].append(off)
        n+=1
    print(f"processed {n} (view,time) pairs\n")

    def summarize(store, label):
        ks = sorted(store.keys())
        print(f"=== by {label} ===")
        print(f"{label:>6} {'PSNR':>8} {'silhIoU':>9} {'cOff(px)':>9} {'n':>5}")
        arr = []
        for k in ks:
            d=store[k]; pp=np.mean(d['psnr']); io=np.mean(d['iou']); of=np.mean(d['off'])
            arr.append((k,pp,io,of)); print(f"{k:>6.0f} {pp:>8.2f} {io:>9.3f} {of:>9.2f} {len(d['psnr']):>5}")
        return np.array(arr)

    A = summarize(by_elev, "elev")
    print()
    Z = summarize(by_az, "azim")

    e,pe,ie,oe = A.T
    sp,ip_ = np.polyfit(e,pe,1); so,_ = np.polyfit(e,oe,1); si,_ = np.polyfit(e,ie,1)
    print(f"\n[D5] PSNR slope {sp*10:+.2f} dB/10deg | elev0={pe[0]:.2f} elev30={pe[-1]:.2f} drop={pe[0]-pe[-1]:.2f} dB")
    print(f"[D6] centroid-offset slope {so:+.2f} px/deg ({so*30:+.1f} px over 0->30) | IoU slope {si*10:+.3f}/10deg")

    fig, ax = plt.subplots(1,3,figsize=(15,4.2))
    ax[0].plot(e,pe,"o-",color="#b85450",lw=2,ms=7)
    ax[0].plot(e,sp*e+ip_,"--",color="gray",alpha=.6,label=f"{sp*10:.2f} dB/10°")
    ax[0].set_xlabel("elevation from input view (deg)"); ax[0].set_ylabel("PSNR SV4D vs clean GT (dB)")
    ax[0].set_title("D5 · VGM spatial fidelity vs angle"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(e,oe,"o-",color="#82b366",lw=2,ms=7)
    ax[1].set_xlabel("elevation (deg)"); ax[1].set_ylabel("centroid offset vs canonical (px)")
    ax[1].set_title("D6 · VGM pose drift vs angle"); ax[1].grid(alpha=.3)
    ax[2].plot(e,ie,"o-",color="#6c8ebf",lw=2,ms=7)
    ax[2].set_xlabel("elevation (deg)"); ax[2].set_ylabel("silhouette IoU vs canonical")
    ax[2].set_title("D6 · Silhouette agreement vs angle"); ax[2].grid(alpha=.3)
    plt.tight_layout()
    out = OUT_FIG/"vgm_inconsistency_curves.png"
    plt.savefig(out,dpi=120,bbox_inches="tight"); print(f"\nsaved {out}")

if __name__=="__main__": main()
