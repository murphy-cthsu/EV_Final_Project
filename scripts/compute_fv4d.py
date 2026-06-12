"""FVD-F / FVD-V / FVD-Diag / FV4D (SV4D-paper style) for SV4D output vs clean d-3dgs.
Grid = V views x T frames. Videos per ordering:
  FVD-F   : per view, scan frames            -> V videos of len T
  FVD-V   : per frame, scan views            -> T videos of len V
  FVD-Diag: walk (v+i, t+i), wrap views      -> V videos of len T
  FV4D    : bidirectional raster over grid   -> chunks of len T
I3D torchscript (StyleGAN-V port). Small-N caveat: report for relative comparison only.
"""
import json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

REPO=Path("/home/cthsu/EV_Final_Project"); T=21
I3D="/mnt/HDD_1/cthsu/metrics_assets/i3d_torchscript.pt"

def loadimg(p,sz=224):
    im=Image.open(p)
    if im.mode=="RGBA":
        a=np.asarray(im.split()[3].resize((sz,sz)),dtype=np.uint8)>127
        bg=Image.new("RGB",im.size,(255,255,255)); bg.paste(im,mask=im.split()[3]); im=bg
        rgb=np.asarray(im.resize((sz,sz)),dtype=np.uint8)
    else:
        im=im.convert("RGB"); rgb=np.asarray(im.resize((sz,sz)),dtype=np.uint8)
        a=(rgb.astype(np.float32)/255.0<0.97).any(-1)
    return rgb,a

def fvd(fa,fb):
    mu_a,mu_b=fa.mean(0),fb.mean(0)
    ca=np.cov(fa,rowvar=False); cb=np.cov(fb,rowvar=False)
    from scipy.linalg import sqrtm
    cs,_=sqrtm(ca@cb,disp=False); cs=cs.real
    return float(((mu_a-mu_b)**2).sum()+np.trace(ca+cb-2*cs))

@torch.no_grad()
def i3d_feats(detector,vids,dev,bs=8):
    out=[]
    for i in range(0,len(vids),bs):
        b=np.stack(vids[i:i+bs])                     # (B,Tc,H,W,3) uint8
        x=torch.from_numpy(b).permute(0,4,1,2,3).contiguous().float().to(dev)
        out.append(detector(x,rescale=True,resize=False,return_features=True).cpu().numpy())
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene",required=True)
    ap.add_argument("--mask_baseplate",action="store_true",
                    help="whiten d3-only persistent content (baseplate) so FVD measures consistency not content difference")
    a=ap.parse_args()
    SCENE=REPO/"data/custom"/a.scene; D3=REPO/"outputs/custom"/f"{a.scene}_d3dgs_ref"/"renders"
    meta=json.loads((SCENE/"transforms_train.json").read_text())
    frames=meta["frames"]+json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    views=sorted({f["view_idx"] for f in frames}); V=len(views)
    print(f"[{a.scene}] grid {V}x{T}; loading images...",flush=True)
    sv=np.zeros((V,T,224,224,3),np.uint8); d3=np.zeros_like(sv); ok=np.zeros((V,T),bool)
    sva=np.zeros((V,T,224,224),bool); d3a=np.zeros_like(sva)
    for v in views:
        for t in range(T):
            flat=v*T+t
            svp=None
            for sp in ("train","test"):
                c=SCENE/sp/f"r_{flat:05d}.png"
                if c.exists(): svp=c; break
            d3p=D3/f"{flat:05d}.png"
            if svp is None or not d3p.exists(): continue
            sv[v,t],sva[v,t]=loadimg(svp); d3[v,t],d3a[v,t]=loadimg(d3p); ok[v,t]=True
    assert ok.all(), f"missing {int((~ok).sum())} frames"
    if a.mask_baseplate:
        # per-view persistent d3-only region (baseplate): d3-FG in >=90% frames, sv-FG in <=10%
        masked_px=0
        for v in range(V):
            base=(d3a[v].mean(0)>=0.9)&(sva[v].mean(0)<=0.1)
            d3[v][:,base]=255
            masked_px+=int(base.sum())
        print(f"masked baseplate: avg {masked_px/V/(224*224)*100:.1f}% of frame per view")
    dev="cuda"; detector=torch.jit.load(I3D).eval().to(dev)
    def orderings(G):
        vids={}
        vids["FVD-F"]=[G[v] for v in range(V)]                                  # V x (T,...)
        vids["FVD-V"]=[G[:,t] for t in range(T)]                                # T x (V,...)
        vids["FVD-Diag"]=[np.stack([G[(v0+i)%V,i] for i in range(T)]) for v0 in range(V)]
        flat=[]
        for v in range(V): flat.extend(G[v] if v%2==0 else G[v,::-1])           # zig-zag raster
        flat=np.stack(flat); vids["FV4D"]=[flat[i:i+T] for i in range(0,len(flat)-T+1,T)]
        return vids
    Osv,Od3=orderings(sv),orderings(d3)
    print(f"{'metric':>9} {'FVD(SV4D vs clean)':>20}  n_videos")
    res={}
    for k in ["FVD-F","FVD-V","FVD-Diag","FV4D"]:
        fa=i3d_feats(detector,Osv[k],dev); fb=i3d_feats(detector,Od3[k],dev)
        res[k]=fvd(fa,fb)
        print(f"{k:>9} {res[k]:>20.1f}  {len(Osv[k])}")
    np.savez(REPO/f"runs_aux/fv4d_{a.scene}.npz",**res)
    print("saved npz. NOTE: small-N FVD — use for relative comparison only.")
if __name__=="__main__": main()
