"""Capacity-controlled fit-residual inconsistency probe (lego_v2, registered canonical).

R_vgm(v,t)   = |render(modelA) - SV4D|   modelA fit to SV4D  -> can't reproduce inconsistent views
R_clean(v,t) = |render(modelB) - d3dgs|  modelB fit to clean  -> model-capacity floor
gap = R_vgm - R_clean  (same model capacity; difference = VGM inconsistency, not model limitation)

Per-view (azimuth) + per-pixel localization map. Uses OUR method as the instrument.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO=Path("/home/cthsu/EV_Final_Project"); sys.path.insert(0,str(REPO/"third_party"/"SC-GS"))
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera as SCGSCamera
from gaussian_renderer import render
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal

SCENE=REPO/"data/custom/lego_v2"; D3=REPO/"outputs/custom/lego_v2_d3dgs_ref/renders"
CANON=REPO/"outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply"
H=W=576; T=21

def aa2mat(aa):
    th=np.linalg.norm(aa,axis=-1,keepdims=True).clip(min=1e-8); ax=aa/th
    K=np.zeros((aa.shape[0],3,3))
    K[:,0,1]=-ax[:,2];K[:,0,2]=ax[:,1];K[:,1,0]=ax[:,2];K[:,1,2]=-ax[:,0];K[:,2,0]=-ax[:,1];K[:,2,1]=ax[:,0]
    I=np.eye(3)[None].repeat(aa.shape[0],0)
    return I+np.sin(th[...,None])*K+(1-np.cos(th[...,None]))*(K@K)

def load(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4: a=im[...,3:4]; rgb=im[...,:3]*a+(1-a)
    else: rgb=im[...,:3]
    return rgb

def render_model(label, g, xyz_canon, frames, fov_x, pipe, bg, FovY):
    s=np.load(REPO/f"outputs/custom/partrigid_{label}/partrigid_state.npz",allow_pickle=True)
    trans=s["trans"];aa=s["aa"];centers=s["arm_centers"];lbs=s["lbs_weights"]
    K,T_train,_=trans.shape; N=xyz_canon.shape[0]
    scale=s["scale"] if "scale" in s.files else None
    xyzr=s["xyz_residual"] if "xyz_residual" in s.files else None
    aidx=s["arm_idx_for_residual"] if "arm_idx_for_residual" in s.files else None
    out={}
    for f in frames:
        v=int(f["view_idx"]);t=int(f["frame_idx"]);tl=min(t,T_train-1)
        c2w=np.asarray(f["transform_matrix"],float);M=np.linalg.inv(c2w)
        Rc=-np.transpose(M[:3,:3]);Rc[:,0]=-Rc[:,0];Tc=-M[:3,3]
        R_all=aa2mat(aa[:,tl,:]);T_all=trans[:,tl,:]
        rel=xyz_canon[:,None,:]-centers[None,:,:]
        rot=np.einsum("kij,nkj->nki",R_all,rel)
        nperp=rot+centers[None,:,:]+T_all[None,:,:]
        wsum=(lbs[...,None]*nperp).sum(1); wt=lbs.sum(1,keepdims=True).clip(0,1)
        nx=wsum+(1-wt)*xyz_canon
        if xyzr is not None and aidx is not None and len(aidx)>0: nx[aidx]=nx[aidx]+xyzr[:,tl,:]
        d_xyz=torch.from_numpy((nx-xyz_canon).astype(np.float32)).cuda()
        d_rot=torch.zeros(N,4,device="cuda")-torch.tensor([1,0,0,0],device="cuda")
        d_sc=torch.zeros(N,3,device="cuda")
        if scale is not None: d_sc=torch.from_numpy((lbs@scale[:,tl,:]).astype(np.float32)).cuda()
        cam=SCGSCamera(colmap_id=0,R=Rc,T=Tc,FoVx=fov_x,FoVy=FovY,image=torch.zeros(3,H,W).cuda(),
                       gt_alpha_mask=None,image_name="x",uid=0,fid=torch.tensor(0.).float())
        with torch.no_grad():
            pkg=render(cam,g,pipe,bg,d_xyz=d_xyz,d_rotation=d_rot,d_scaling=d_sc,d_rot_as_res=True)
        out[(v,t)]=torch.clamp(pkg["render"],0,1).cpu().numpy().transpose(1,2,0)
    return out

def main():
    meta=json.loads((SCENE/"transforms_train.json").read_text())
    frames=meta["frames"]+json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    fov_x=meta["camera_angle_x"]; FovY=focal2fov(fov2focal(fov_x,W),H)
    pp=PipelineParams(_A()); pipe=pp.extract(_A().parse_args([])) if False else pp.extract(__import__("argparse").Namespace())
    # simpler pipe
    pa=_A(); pipe=PipelineParams(pa).extract(pa.parse_args([]))
    bg=torch.tensor([1,1,1],dtype=torch.float32,device="cuda")
    g=GaussianModel(3,fea_dim=8,with_motion_mask=False); g.load_ply(str(CANON),og_number_points=0)
    xyz=g.get_xyz.detach().cpu().numpy()
    # azimuth per view (lego_v2: 60deg gaps)
    v2az={0:0,1:60,2:120,3:180,4:240}

    print("rendering Model A (fit to SV4D)...")
    rA=render_model("lego_v2_A1_leakfree_B",g,xyz,frames,fov_x,pipe,bg,FovY)
    print("rendering Model B (fit to clean d-3dgs)...")
    rB=render_model("lego_v2_d3dgs_sup_ceiling",g,xyz,frames,fov_x,pipe,bg,FovY)

    by_az=defaultdict(lambda:{"rvgm":[],"rclean":[]})
    pixmap=defaultdict(lambda:{"vgm":np.zeros((H,W)),"clean":np.zeros((H,W)),"n":0})
    for f in frames:
        v=int(f["view_idx"]);t=int(f["frame_idx"]);flat=v*T+t
        sv=load(SCENE/("train" if (SCENE/"train"/f"r_{flat:05d}.png").exists() else "test")/f"r_{flat:05d}.png")
        d3=load(D3/f"{flat:05d}.png")
        rvgm=np.abs(rA[(v,t)]-sv).mean(-1)     # modelA vs SV4D
        rclean=np.abs(rB[(v,t)]-d3).mean(-1)   # modelB vs clean
        by_az[v2az[v]]["rvgm"].append(float(rvgm.mean()))
        by_az[v2az[v]]["rclean"].append(float(rclean.mean()))
        pm=pixmap[v]; pm["vgm"]+=rvgm; pm["clean"]+=rclean; pm["n"]+=1

    azs=sorted(by_az)
    Rv=np.array([np.mean(by_az[a]["rvgm"]) for a in azs])
    Rc=np.array([np.mean(by_az[a]["rclean"]) for a in azs])
    gap=Rv-Rc
    print(f"\n{'azim':>5} {'R_vgm':>8} {'R_clean':>8} {'gap':>8}")
    for a,rv,rc,g_ in zip(azs,Rv,Rc,gap): print(f"{a:>5} {rv:>8.4f} {rc:>8.4f} {g_:>8.4f}")
    print(f"\ngap mean={gap.mean():.4f}  range={gap.max()-gap.min():.4f}")
    print(f"R_clean range={Rc.max()-Rc.min():.4f} (should be SMALL+uniform if capacity is view-independent)")
    print(f"gap/Rclean ratio = {gap.mean()/max(Rc.mean(),1e-9):.2f}x  (inconsistency vs model floor)")

    # plot: gap vs azimuth + per-pixel gap map for view 0
    fig=plt.figure(figsize=(13,4.3))
    ax1=fig.add_subplot(1,3,1)
    ax1.plot(azs,Rv,"o-",color="#b85450",label="R_vgm (fit→SV4D)")
    ax1.plot(azs,Rc,"s-",color="#6c8ebf",label="R_clean (fit→clean, floor)")
    ax1.fill_between(azs,Rc,Rv,alpha=.15,color="#d6840b",label="gap = inconsistency")
    ax1.set_xlabel("azimuth from input (deg)"); ax1.set_ylabel("mean fit residual (L1)")
    ax1.set_title("Capacity-controlled fit residual"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
    ax2=fig.add_subplot(1,3,2)
    ax2.plot(azs,gap,"o-",color="#d6840b",lw=2,ms=7)
    ax2.set_xlabel("azimuth from input (deg)"); ax2.set_ylabel("gap (R_vgm - R_clean)")
    ax2.set_title("Inconsistency vs viewpoint (gap)"); ax2.grid(alpha=.3)
    # per-pixel gap map for worst-azimuth view
    worst_v=[v for v,a in v2az.items() if a==azs[int(gap.argmax())]][0]
    pm=pixmap[worst_v]; gmap=(pm["vgm"]-pm["clean"])/pm["n"]
    ax3=fig.add_subplot(1,3,3)
    im=ax3.imshow(np.clip(gmap,0,None),cmap="magma"); ax3.axis("off")
    ax3.set_title(f"Inconsistency map, az={azs[int(gap.argmax())]}° (gap per px)")
    fig.colorbar(im,ax=ax3,fraction=.046)
    plt.tight_layout(); out=REPO/"meetTW_checkpoint_0601/figs/fit_residual_probe.png"
    plt.savefig(out,dpi=120,bbox_inches="tight"); print(f"saved {out}")

if __name__=="__main__":
    import torch; main()
