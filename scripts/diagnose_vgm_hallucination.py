"""D2 + D7: WHERE does SV4D hallucinate (spatial localization)?

D2 boundary-vs-interior: residual |SV4D - clean| as a function of distance from the
   silhouette boundary. Tests whether hallucination concentrates at edges or interior.
D7 hallucination type: decompose error into INVENTED (sv has content, clean doesn't,
   excl. baseplate), MISSED (clean has, sv doesn't), WRONG (both have, differ).
Also saves an aggregate residual heatmap for one representative far-side view.
"""
import json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path("/home/cthsu/EV_Final_Project")
OUT=REPO/"meetTW_checkpoint_0601/figs"; T=21

def load(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4: a=im[...,3]; rgb=im[...,:3]*a[...,None]+(1-a[...,None])
    else: rgb=im[...,:3]; a=(rgb<0.97).any(-1).astype(np.float32)
    return rgb,a

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scene",default="lego_v3"); a=ap.parse_args()
    SCENE=REPO/"data/custom"/a.scene; D3=REPO/"outputs/custom"/f"{a.scene}_d3dgs_ref"/"renders"
    meta=json.loads((SCENE/"transforms_train.json").read_text())["frames"]+\
         json.loads((SCENE/"transforms_test.json").read_text())["frames"]

    # distance-from-boundary bins (px): 0-2(edge),2-5,5-10,10-20,20+
    bins=[0,2,5,10,20,9999]; blabels=["0-2","2-5","5-10","10-20","20+"]
    res_by_bin=defaultdict(list)
    inv=mis=wrong=0.0; tot=0.0
    n=0
    for f in meta:
        v,t=f["view_idx"],f["frame_idx"]; flat=v*T+t
        sv=None
        for sp in ("train","test"):
            c=SCENE/sp/f"r_{flat:05d}.png"
            if c.exists(): sv=c; break
        d3p=D3/f"{flat:05d}.png"
        if sv is None or not d3p.exists(): continue
        svr,sva=load(sv); d3r,d3a=load(d3p)
        svf,d3f=sva>.5,d3a>.5
        base=d3f&~svf                       # baseplate / clean-only structure
        resid=np.abs(svr-d3r).mean(-1)
        # D2: within the SV4D foreground (object), residual vs distance to its boundary
        if svf.sum()>50:
            dist=distance_transform_edt(svf)   # px from boundary, inside object
            for lo,hi,lab in zip(bins[:-1],bins[1:],blabels):
                m=svf&(dist>=lo)&(dist<hi)
                if m.sum()>0: res_by_bin[lab].append(float(resid[m].mean()))
        # D7: error-energy decomposition (exclude baseplate region from 'missed')
        e=resid**2
        invented=(svf&~d3f)                  # sv content where clean empty = hallucinated
        missed=(d3f&~svf)&~base              # clean object sv missed (base excluded as artifact)
        wrong_m=(svf&d3f)                    # both present, intensity differs
        inv+=float(e[invented].sum()); mis+=float(e[missed].sum())
        wrong+=float(e[wrong_m].sum()); tot+=float(e[svf|((d3f)&~base)].sum())
        n+=1
    print(f"[{a.scene}] processed {n} frames")

    # D2 summary
    labs=blabels; means=[np.mean(res_by_bin[l]) if res_by_bin[l] else 0 for l in labs]
    print(f"\n[D2 {a.scene}] residual vs distance-from-boundary (px):")
    for l,m in zip(labs,means): print(f"   {l:>6} px : {m:.4f}")
    print(f"   edge/interior ratio (0-2 / 20+) = {means[0]/max(means[-1],1e-9):.2f}x")

    # D7 summary
    s=inv+mis+wrong
    print(f"\n[D7 {a.scene}] error-energy decomposition:")
    print(f"   INVENTED (hallucinated content) : {inv/s*100:5.1f}%")
    print(f"   MISSED   (clean object dropped) : {mis/s*100:5.1f}%")
    print(f"   WRONG    (both present, differ) : {wrong/s*100:5.1f}%")

    # plot
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    ax[0].bar(labs,means,color="#b85450",alpha=.85)
    ax[0].set_xlabel("distance from silhouette boundary (px)"); ax[0].set_ylabel("mean |SV4D - clean| residual")
    ax[0].set_title(f"D2 · hallucination vs boundary distance ({a.scene})\nedge is {means[0]/max(means[-1],1e-9):.1f}× the interior")
    ax[0].grid(axis="y",alpha=.3)
    ax[1].pie([inv,mis,wrong],labels=[f"invented\n{inv/s*100:.0f}%",f"missed\n{mis/s*100:.0f}%",f"wrong\n{wrong/s*100:.0f}%"],
              colors=["#d6840b","#6c8ebf","#b85450"],autopct="",startangle=90,wedgeprops=dict(width=.45))
    ax[1].set_title(f"D7 · error-energy type ({a.scene})")
    plt.tight_layout(); out=OUT/f"vgm_hallucination_{a.scene}.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
    print(f"\nsaved {out}")

if __name__=="__main__": main()
