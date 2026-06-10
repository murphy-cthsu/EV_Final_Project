"""D3: temporal flicker of SV4D in provably-static regions.

A pixel is 'provably static' if the clean reference (d-3dgs) has near-zero temporal
variance there (the true scene doesn't move). Any temporal variance SV4D shows in
those pixels is hallucinated flicker — VGM temporal inconsistency, not real motion.

Outputs: per-view flicker energy, binned by azimuth/elevation; separates 'real motion'
(d3 moves) from 'flicker' (d3 static, sv4d moves). Pure VGM characterization.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/cthsu/EV_Final_Project")
SCENE = REPO/"data/custom/lego_v3"; D3=REPO/"outputs/custom/lego_v3_d3dgs_ref/renders"
OUT=REPO/"meetTW_checkpoint_0601/figs"
T=21
STATIC_THR = 0.015   # d3 temporal std below this = provably static (0-1 intensity)

def load_gray_a(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4: a=im[...,3]; rgb=im[...,:3]*a[...,None]+(1-a[...,None])
    else: rgb=im[...,:3]; a=(rgb<0.97).any(-1).astype(np.float32)
    return rgb.mean(-1), a   # grayscale, alpha

def main():
    meta=json.loads((SCENE/"transforms_train.json").read_text())["frames"]+\
         json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    v2e={f["view_idx"]:f["elevation_deg"] for f in meta}
    v2a={f["view_idx"]:f["azimuth_deg"] for f in meta}
    views=sorted({f["view_idx"] for f in meta})

    be=defaultdict(lambda:defaultdict(list)); ba=defaultdict(lambda:defaultdict(list))
    rows=[]
    for v in views:
        sv_stack=[]; d3_stack=[]; a_stack=[]
        ok=True
        for t in range(T):
            flat=v*T+t
            svp=None
            for sp in ("train","test"):
                c=SCENE/sp/f"r_{flat:05d}.png"
                if c.exists(): svp=c; break
            d3p=D3/f"{flat:05d}.png"
            if svp is None or not d3p.exists(): ok=False; break
            sg,sa=load_gray_a(svp); dg,da=load_gray_a(d3p)
            sv_stack.append(sg); d3_stack.append(dg); a_stack.append((sa>.5)&(da>.5))
        if not ok: continue
        sv=np.stack(sv_stack); d3=np.stack(d3_stack); fg=np.stack(a_stack).any(0)
        sv_std=sv.std(0); d3_std=d3.std(0)
        static = (d3_std < STATIC_THR) & fg        # provably static (clean ref still)
        moving = (d3_std >= STATIC_THR) & fg       # real motion region
        if static.sum()<50: continue
        flicker = float(sv_std[static].mean())            # SV4D variance where scene is static
        realmot = float(sv_std[moving].mean()) if moving.sum()>0 else 0.0
        # spurious-motion area: fraction of static pixels SV4D treats as moving
        spurious = float((sv_std[static] > STATIC_THR).mean())
        rows.append((v,v2e[v],v2a[v],flicker,realmot,spurious))
        for st,k in ((be,v2e[v]),(ba,v2a[v])):
            st[k]["flick"].append(flicker); st[k]["spur"].append(spurious); st[k]["real"].append(realmot)

    # overall
    fl=np.array([r[3] for r in rows]); rm=np.array([r[4] for r in rows]); sp=np.array([r[5] for r in rows])
    print(f"views processed: {len(rows)}")
    print(f"\n=== overall (static_thr={STATIC_THR}) ===")
    print(f"mean flicker (SV4D std in provably-static px) : {fl.mean():.4f}")
    print(f"mean SV4D std in real-motion px               : {rm.mean():.4f}")
    print(f"flicker / real-motion ratio                   : {fl.mean()/max(rm.mean(),1e-9):.2f}")
    print(f"spurious-motion area (frac of static px SV4D flags moving): {sp.mean():.1%}")

    def summ(store,label):
        ks=sorted(store); print(f"\n=== flicker by {label} ===")
        print(f"{label:>6} {'flicker':>9} {'spurious%':>10} {'n':>4}")
        out=[]
        for k in ks:
            d=store[k]; f=np.mean(d['flick']); s=np.mean(d['spur'])
            out.append((k,f,s)); print(f"{k:>6.0f} {f:>9.4f} {s:>9.1%} {len(d['flick']):>4}")
        return np.array(out)
    E=summ(be,"elev"); Z=summ(ba,"azim")

    # plot: flicker vs azimuth (polar) + flicker vs elevation
    fig=plt.figure(figsize=(10,4.3))
    az=Z[:,0]; fz=Z[:,1]
    ax1=fig.add_subplot(1,2,1,projection="polar")
    ang=np.deg2rad(np.append(az,az[0])); val=np.append(fz,fz[0])
    ax1.plot(ang,val,"o-",color="#d6840b",lw=2,ms=6); ax1.fill(ang,val,color="#d6840b",alpha=.12)
    ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
    ax1.set_title("D3 · VGM temporal flicker vs azimuth\n(SV4D std in static px; input at 0°)",pad=18)
    ax2=fig.add_subplot(1,2,2)
    el=E[:,0]; fe=E[:,1]; s,i=np.polyfit(el,fe,1)
    ax2.plot(el,fe,"o-",color="#d6840b",lw=2,ms=7)
    ax2.plot(el,s*el+i,"--",color="gray",alpha=.6,label=f"{s*10:+.4f}/10°")
    ax2.set_xlabel("elevation from input (deg)"); ax2.set_ylabel("flicker (SV4D temporal std, static px)")
    ax2.set_title("D3 · temporal flicker vs elevation"); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout(); out=OUT/"vgm_temporal_flicker.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
    print(f"\nsaved {out}")
    print(f"azimuth flicker: input(0°)={fz[0]:.4f}, max={fz.max():.4f}@az{az[int(fz.argmax())]:.0f}, ratio={fz.max()/fz[0]:.1f}x")

if __name__=="__main__": main()
