"""J3: prove the reliability cone can be measured with a STATIC canonical only
(no 4D GT / d-3dgs).

Cone-A (canonical-only): PSNR(SV4D, canonical_render) inside the projected
  static-BODY mask (part_id==1), per azimuth. Body doesn't move -> canonical IS
  absolute GT there -> deviation = pure VGM inconsistency. No d-3dgs.
Cone-B (d-3dgs): PSNR(SV4D, d-3dgs) baseplate-excluded, per azimuth (what we had).

If Cone-A and Cone-B share the SHAPE (peak at input, far-side dip, high corr),
the static canonical is a sufficient instrument -> J3 resolved, GT-free claim proven.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from collections import defaultdict

REPO=Path("/home/cthsu/EV_Final_Project"); T=21
SCENE=REPO/"data/custom/lego_v3"; D3=REPO/"outputs/custom/lego_v3_d3dgs_ref/renders"
CANON_RENDER=REPO/"outputs/custom/lego_v3_canon_static_render"
PLY=REPO/"outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"
PART=REPO/"runs_aux/part_assignment_lego_v3/part_id.npy"
sys.path.insert(0,str(REPO/"third_party"/"SC-GS"))
from scene.gaussian_model import GaussianModel

def load(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4: a=im[...,3]; rgb=im[...,:3]*a[...,None]+(1-a[...,None])
    else: rgb=im[...,:3]; a=(rgb<0.97).any(-1).astype(np.float32)
    return rgb,a

def project(xyz,c2w,fov_x,H,W):
    w2c=np.linalg.inv(c2w); flip=np.diag([1.,-1.,-1.,1.]); w2c=flip@w2c
    xh=np.concatenate([xyz,np.ones((len(xyz),1))],1); cam=(w2c@xh.T).T[:,:3]
    z=cam[:,2]; fx=(W/2)/np.tan(fov_x/2); valid=z>0
    u=fx*cam[:,0]/np.maximum(z,1e-6)+W/2; v=fx*cam[:,1]/np.maximum(z,1e-6)+H/2
    return u,v,valid

def main():
    H=W=576
    meta=json.loads((SCENE/"transforms_train.json").read_text())["frames"]+\
         json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    v2a={f["view_idx"]:f["azimuth_deg"] for f in meta}
    cams={}
    fov_x=json.loads((SCENE/"transforms_train.json").read_text())["camera_angle_x"]
    for f in meta:
        cams.setdefault(f["view_idx"],np.asarray(f["transform_matrix"],float))
    g=GaussianModel(3,fea_dim=0,with_motion_mask=False); g.load_ply(str(PLY),og_number_points=0)
    xyz=g.get_xyz.detach().cpu().numpy()
    pid=np.load(PART); body=xyz[pid==1]
    print(f"body gaussians: {len(body)}")

    # build static-body mask per view
    body_mask={}
    for v,c2w in cams.items():
        u,vv,val=project(body,c2w,fov_x,H,W)
        m=np.zeros((H,W),bool)
        ui=np.clip(u[val].astype(int),0,W-1); vi=np.clip(vv[val].astype(int),0,H-1)
        m[vi,ui]=True
        m=binary_dilation(m,iterations=3)
        m=binary_erosion(m,iterations=1)  # clean
        body_mask[v]=m

    baz=defaultdict(lambda:defaultdict(list))
    for f in meta:
        v,t=f["view_idx"],f["frame_idx"]; flat=v*T+t
        svp=None
        for sp in ("train","test"):
            c=SCENE/sp/f"r_{flat:05d}.png"
            if c.exists(): svp=c; break
        d3p=D3/f"{flat:05d}.png"; cnp=CANON_RENDER/f"{flat:05d}.png"
        if svp is None or not d3p.exists() or not cnp.exists(): continue
        svr,sva=load(svp); d3r,d3a=load(d3p); cnr,cna=load(cnp)
        # Cone-A: SV4D vs canonical, in static body mask
        bm=body_mask[v] & (cna>0.5)
        if bm.sum()>50:
            mseA=(((svr-cnr)**2).mean(-1)*bm).sum()/bm.sum()
            baz[v2a[v]]["A"].append(-10*np.log10(max(mseA,1e-12)))
        # Cone-B: SV4D vs d-3dgs, baseplate-excluded
        svf,d3f=sva>.5,d3a>.5; keep=~(d3f&~svf)
        mseB=(((svr-d3r)**2).mean(-1)*keep).sum()/max(keep.sum(),1)
        baz[v2a[v]]["B"].append(-10*np.log10(max(mseB,1e-12)))

    azs=sorted(baz)
    A=np.array([np.mean(baz[k]["A"]) for k in azs])
    B=np.array([np.mean(baz[k]["B"]) for k in azs])
    azn=np.array(azs)
    print(f"\n{'azim':>6} {'Cone-A (canon-only)':>20} {'Cone-B (d-3dgs)':>18}")
    for k,a,b in zip(azs,A,B): print(f"{k:>6.0f} {a:>20.2f} {b:>18.2f}")
    # shape agreement
    corr=np.corrcoef(A,B)[0,1]
    # rank correlation
    from scipy.stats import spearmanr
    rho,_=spearmanr(A,B)
    print(f"\nPearson corr(A,B) = {corr:.3f}   Spearman rho = {rho:.3f}")
    print(f"Cone-A: peak az={azn[A.argmax()]:.0f}({A.max():.1f})  trough az={azn[A.argmin()]:.0f}({A.min():.1f})  range={A.max()-A.min():.1f}")
    print(f"Cone-B: peak az={azn[B.argmax()]:.0f}({B.max():.1f})  trough az={azn[B.argmin()]:.0f}({B.min():.1f})  range={B.max()-B.min():.1f}")

    # plot overlay (normalized, since refs differ)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def nrm(x): return (x-x.min())/(x.max()-x.min()+1e-9)
    fig=plt.figure(figsize=(11,4.6))
    ax=fig.add_subplot(1,2,1,projection="polar")
    a=np.deg2rad(np.append(azn,azn[0]))
    ax.plot(a,np.append(nrm(A),nrm(A)[0]),"o-",color="#9673a6",lw=2,ms=5,label="Cone-A: canonical-only (no GT)")
    ax.plot(a,np.append(nrm(B),nrm(B)[0]),"s--",color="#b85450",lw=2,ms=5,alpha=.7,label="Cone-B: d-3dgs (4D GT)")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title("Reliability cone: canonical-only vs d-3dgs\n(normalized; input at 0°)",pad=22,fontsize=10)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.15),fontsize=8)
    ax2=fig.add_subplot(1,2,2)
    ax2.scatter(B,A,c=azn,cmap="twilight",s=60)
    for k,a_,b_ in zip(azs,A,B): ax2.annotate(f"{int(k)}",(b_,a_),fontsize=7)
    ax2.set_xlabel("Cone-B PSNR vs d-3dgs (dB)"); ax2.set_ylabel("Cone-A PSNR vs canonical, body (dB)")
    ax2.set_title(f"Agreement: Pearson={corr:.2f}, Spearman={rho:.2f}"); ax2.grid(alpha=.3)
    plt.tight_layout(); out=REPO/"meetTW_checkpoint_0601/figs/j3_canonical_only_cone.png"
    plt.savefig(out,dpi=120,bbox_inches="tight"); print(f"saved {out}")

if __name__=="__main__": main()
