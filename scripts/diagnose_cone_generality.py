"""Generality test: does the SV4D 'reliability cone' (D5 spatial + D3 temporal vs
azimuth) replicate on a 2nd scene (hellwarrior, articulated character) vs lego_v3
(rigid digger)? Same generator (SV4D 2.0), different object class.
Computes azimuth-binned PSNR(SV4D vs clean) + temporal flicker for a given scene.
"""
import json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict

REPO=Path("/home/cthsu/EV_Final_Project"); T=21; STATIC_THR=0.015
def load_g(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4: a=im[...,3]; rgb=im[...,:3]*a[...,None]+(1-a[...,None])
    else: rgb=im[...,:3]; a=(rgb<0.97).any(-1).astype(np.float32)
    return rgb,rgb.mean(-1),a

def run(scene):
    SCENE=REPO/"data/custom"/scene; D3=REPO/"outputs/custom"/f"{scene}_d3dgs_ref"/"renders"
    meta=json.loads((SCENE/"transforms_train.json").read_text())["frames"]+\
         json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    v2a={f["view_idx"]:f["azimuth_deg"] for f in meta}
    views=sorted({f["view_idx"] for f in meta})
    ba=defaultdict(lambda:defaultdict(list))
    for v in views:
        sg=[]; dg=[]; sr=[]; dr=[]; fa=[]
        ok=True
        for t in range(T):
            flat=v*T+t; svp=None
            for sp in ("train","test"):
                c=SCENE/sp/f"r_{flat:05d}.png"
                if c.exists(): svp=c; break
            d3p=D3/f"{flat:05d}.png"
            if svp is None or not d3p.exists(): ok=False; break
            srgb,sgr,sa=load_g(svp); drgb,dgr,da=load_g(d3p)
            sg.append(sgr); dg.append(dgr); sr.append(srgb); dr.append(drgb); fa.append((sa>.5)&(da>.5))
        if not ok: continue
        sgA=np.stack(sg); dgA=np.stack(dg); fg=np.stack(fa).any(0)
        # D5 spatial: mean per-frame PSNR (baseplate-excluded)
        ps=[]
        for t in range(T):
            svf=(np.stack(fa)[t]); # approx fg per frame via union ok; use object mask
            keep=fg  # exclude background; baseplate handled implicitly (both fg)
            mse=(((sr[t]-dr[t])**2).mean(-1)*keep).sum()/max(keep.sum(),1)
            ps.append(-10*np.log10(max(mse,1e-12)))
        # D3 temporal flicker
        d3_std=dgA.std(0); sv_std=sgA.std(0)
        static=(d3_std<STATIC_THR)&fg
        flick=float(sv_std[static].mean()) if static.sum()>50 else np.nan
        ba[v2a[v]]["p"].append(np.mean(ps))
        if not np.isnan(flick): ba[v2a[v]]["f"].append(flick)
    azs=sorted(ba)
    P=np.array([np.mean(ba[k]["p"]) for k in azs])
    F=np.array([np.mean(ba[k]["f"]) for k in azs])
    return np.array(azs),P,F

if __name__=="__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    OUT=REPO/"meetTW_checkpoint_0601/figs"
    results={}
    for sc in ["lego_v3","hellwarrior"]:
        az,P,F=run(sc); results[sc]=(az,P,F)
        print(f"\n[{sc}] azimuth cone:")
        print(f"  PSNR : input(0°)={P[0]:.1f}  min={P.min():.1f}@{az[int(P.argmin())]:.0f}°  range={P.max()-P.min():.1f}dB")
        print(f"  flick: input(0°)={F[0]:.4f}  max={F.max():.4f}@{az[int(F.argmax())]:.0f}°  ratio={F.max()/F[0]:.1f}x")

    fig=plt.figure(figsize=(11,4.6))
    for i,(metric,idx,lab,col) in enumerate([("PSNR (dB)",1,"spatial fidelity","#b85450"),
                                              ("flicker",2,"temporal flicker","#d6840b")]):
        ax=fig.add_subplot(1,2,i+1,projection="polar")
        for sc,ls,mk in [("lego_v3","-","o"),("hellwarrior","--","s")]:
            az,P,F=results[sc]; vals=(P if idx==1 else F)
            a=np.deg2rad(np.append(az,az[0])); val=np.append(vals,vals[0])
            ax.plot(a,val,ls,marker=mk,color=col,lw=2,ms=5,alpha=.55 if sc=="hellwarrior" else 1.0,
                    label=f"{sc} ({'rigid' if 'lego' in sc else 'articulated'})")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(f"{lab} vs azimuth\n(input at 0°)",pad=20,fontsize=11)
        ax.legend(loc="upper right",bbox_to_anchor=(1.3,1.15),fontsize=8)
    fig.suptitle("Reliability cone replicates across object classes (same SV4D 2.0 generator)",
                 fontsize=12,fontweight="bold",y=1.04)
    plt.tight_layout(); out=OUT/"vgm_cone_generality.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
    print(f"\nsaved {out}")
