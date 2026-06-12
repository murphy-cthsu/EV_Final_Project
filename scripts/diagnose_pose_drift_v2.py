"""D6 v2 — pose drift with a REGISTERED reference (fixes R2 from code review).
Old D6 compared SV4D centroids to canonical-render centroids, but the canonical is
NOT registered to lego_v3 (IoU 0.28) -> contaminated. v2 compares MOTION-REGION
centroids of SV4D vs clean d-3dgs at the same (view,t): weight = |frame - temporal
median| inside FG. Static content (baseplate, body) cancels; the moving part's
position error = pose drift, against a registered reference."""
import json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict

REPO=Path("/home/cthsu/EV_Final_Project"); T=21
def load_gray_fg(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4:
        a=im[...,3]; rgb=im[...,:3]*a[...,None]+(1-a[...,None])
    else:
        rgb=im[...,:3]; a=(rgb<0.97).any(-1).astype(np.float32)
    return rgb.mean(-1), a>0.5

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scene",required=True); a=ap.parse_args()
    SCENE=REPO/"data/custom"/a.scene; D3=REPO/"outputs/custom"/f"{a.scene}_d3dgs_ref"/"renders"
    meta=json.loads((SCENE/"transforms_train.json").read_text())
    frames=meta["frames"]+json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    v2e={f["view_idx"]:f["elevation_deg"] for f in frames}
    v2a={f["view_idx"]:f["azimuth_deg"] for f in frames}
    views=sorted({f["view_idx"] for f in frames})
    be=defaultdict(list); ba=defaultdict(list)
    def cen(w):
        s=w.sum()
        if s<50: return None
        ys,xs=np.nonzero(w); ww=w[ys,xs]
        return np.array([(xs*ww).sum()/ww.sum(),(ys*ww).sum()/ww.sum()])
    for v in views:
        sv_g=[];sv_f=[];d3_g=[];d3_f=[];ok=True
        for t in range(T):
            flat=v*T+t; svp=None
            for sp in ("train","test"):
                c=SCENE/sp/f"r_{flat:05d}.png"
                if c.exists(): svp=c; break
            d3p=D3/f"{flat:05d}.png"
            if svp is None or not d3p.exists(): ok=False; break
            g1,f1=load_gray_fg(svp); g2,f2=load_gray_fg(d3p)
            sv_g.append(g1);sv_f.append(f1);d3_g.append(g2);d3_f.append(f2)
        if not ok: continue
        sv_g=np.stack(sv_g);d3_g=np.stack(d3_g)
        med_sv=np.median(sv_g,0); med_d3=np.median(d3_g,0)
        for t in range(T):
            w_sv=np.abs(sv_g[t]-med_sv)*sv_f[t]
            w_d3=np.abs(d3_g[t]-med_d3)*d3_f[t]
            c1,c2=cen(w_sv),cen(w_d3)
            if c1 is None or c2 is None: continue
            off=float(np.linalg.norm(c1-c2))
            be[v2e[v]].append(off); ba[v2a[v]].append(off)
    es=sorted(be); E=np.array([np.median(be[k]) for k in es])
    azs=sorted(ba); A=np.array([np.median(ba[k]) for k in azs])
    print(f"[{a.scene}] D6-v2 motion-centroid drift vs REGISTERED d-3dgs reference (px, median)")
    print(f"{'elev':>6} "+" ".join(f"{k:>6.0f}" for k in es)); print(f"{'drift':>6} "+" ".join(f"{x:>6.1f}" for x in E))
    s,i=np.polyfit(es,E,1)
    print(f"elev slope: {s:+.2f} px/deg  ({s*30:+.1f} px over 0->30)")
    print(f"{'azim':>6} "+" ".join(f"{k:>6.0f}" for k in azs)); print(f"{'drift':>6} "+" ".join(f"{x:>6.1f}" for x in A))
    print(f"input-az drift={A[0]:.1f}px  max={A.max():.1f}px @az{azs[int(A.argmax())]:.0f}")
    np.savez(REPO/f"runs_aux/pose_drift_v2_{a.scene}.npz",es=es,E=E,azs=azs,A=A)
if __name__=="__main__": main()
