"""Generic capacity-controlled fit-residual VGM-inconsistency probe (GPU-native).
Args: --scene, --canon, --label_vgm (model fit to SV4D), --label_floor (model fit to clean).
gap = R_vgm - R_clean reproduces the inconsistency cone using OUR method as instrument.
Bins by azimuth AND elevation; saves npz + per-pixel localization map for the worst view."""
import json, sys, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
REPO=Path("/home/cthsu/EV_Final_Project"); sys.path.insert(0,str(REPO/"third_party"/"SC-GS"))
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera as SCGSCamera
from gaussian_renderer import render
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal
H=W=576; T=21
def loadimg(p):
    im=np.asarray(Image.open(p),dtype=np.float32)/255.0
    if im.shape[-1]==4: a=im[...,3:4]; return im[...,:3]*a+(1-a)
    return im[...,:3]
def loadstate(label):
    s=np.load(REPO/f"outputs/custom/partrigid_{label}/partrigid_state.npz",allow_pickle=True)
    d=dict(Tt=s["trans"].shape[1])
    d["trans"]=torch.from_numpy(s["trans"]).float().cuda()
    d["aa"]=torch.from_numpy(s["aa"]).float().cuda()
    d["centers"]=torch.from_numpy(s["arm_centers"]).float().cuda()
    d["lbs"]=torch.from_numpy(s["lbs_weights"]).float().cuda()
    d["nz"]=(d["lbs"].sum(1)>1e-4)
    d["scale"]=torch.from_numpy(s["scale"]).float().cuda() if "scale" in s.files else None
    d["xyzr"]=torch.from_numpy(s["xyz_residual"]).float().cuda() if "xyz_residual" in s.files else None
    d["aidx"]=torch.from_numpy(s["arm_idx_for_residual"]).long().cuda() if "arm_idx_for_residual" in s.files else None
    return d
def render_one(st,g,xyz_t,Rc,Tc,t,fov_x,pipe,bg,FovY,N):
    tl=min(t,st["Tt"]-1)
    aa=st["aa"][:,tl,:]; th=aa.norm(dim=-1,keepdim=True).clamp(min=1e-8); ax=aa/th
    K=torch.zeros(aa.shape[0],3,3,device="cuda")
    K[:,0,1]=-ax[:,2];K[:,0,2]=ax[:,1];K[:,1,0]=ax[:,2];K[:,1,2]=-ax[:,0];K[:,2,0]=-ax[:,1];K[:,2,1]=ax[:,0]
    I=torch.eye(3,device="cuda")[None].expand(aa.shape[0],-1,-1)
    R_all=I+torch.sin(th)[...,None]*K+(1-torch.cos(th)[...,None])*(K@K)
    trans=st["trans"][:,tl,:]; nx=xyz_t.clone(); nz=st["nz"]
    xz=xyz_t[nz]; lz=st["lbs"][nz]
    rel=xz[:,None,:]-st["centers"][None,:,:]
    nperp=torch.einsum("kij,mkj->mki",R_all,rel)+st["centers"][None,:,:]+trans[None,:,:]
    nx[nz]=(lz[...,None]*nperp).sum(1)+(1-lz.sum(1,keepdim=True).clamp(0,1))*xz
    if st["xyzr"] is not None and st["aidx"] is not None and len(st["aidx"])>0:
        nx[st["aidx"]]=nx[st["aidx"]]+st["xyzr"][:,tl,:]
    d_xyz=(nx-xyz_t).contiguous()
    d_rot=torch.zeros(N,4,device="cuda")-torch.tensor([1.,0,0,0],device="cuda")
    d_sc=(st["lbs"]@st["scale"][:,tl,:]).contiguous() if st["scale"] is not None else torch.zeros(N,3,device="cuda")
    cam=SCGSCamera(colmap_id=0,R=Rc,T=Tc,FoVx=fov_x,FoVy=FovY,image=torch.zeros(3,H,W).cuda(),
                   gt_alpha_mask=None,image_name="x",uid=0,fid=torch.tensor(0.).float())
    with torch.no_grad():
        pkg=render(cam,g,pipe,bg,d_xyz=d_xyz,d_rotation=d_rot,d_scaling=d_sc,d_rot_as_res=True)
    return torch.clamp(pkg["render"],0,1).cpu().numpy().transpose(1,2,0)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene",required=True); ap.add_argument("--canon",required=True)
    ap.add_argument("--label_vgm",required=True); ap.add_argument("--label_floor",required=True)
    ap.add_argument("--tag",required=True)
    a=ap.parse_args()
    SCENE=REPO/"data/custom"/a.scene; D3=REPO/"outputs/custom"/f"{a.scene}_d3dgs_ref"/"renders"
    meta=json.loads((SCENE/"transforms_train.json").read_text())
    frames=meta["frames"]+json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    fov_x=meta["camera_angle_x"];FovY=focal2fov(fov2focal(fov_x,W),H)
    pa=_A();pipe=PipelineParams(pa).extract(pa.parse_args([]));bg=torch.tensor([1.,1,1],device="cuda")
    g=GaussianModel(3,fea_dim=8,with_motion_mask=False);g.load_ply(a.canon,og_number_points=0)
    xyz_t=g.get_xyz.detach().float().cuda();N=xyz_t.shape[0]
    A=loadstate(a.label_vgm); B=loadstate(a.label_floor)
    v2az={f["view_idx"]:f["azimuth_deg"] for f in frames}; v2el={f["view_idx"]:f["elevation_deg"] for f in frames}
    baz=defaultdict(lambda:{"v":[],"c":[],"raw":[]}); bel=defaultdict(lambda:{"v":[],"c":[],"raw":[]})
    pix=defaultdict(lambda:{"g":np.zeros((H,W)),"n":0}); done=0
    print(f"start {a.tag}",flush=True)
    for f in frames:
        v=int(f["view_idx"]);t=int(f["frame_idx"])
        if t%5!=0: continue
        flat=v*T+t
        svp=SCENE/("train" if (SCENE/"train"/f"r_{flat:05d}.png").exists() else "test")/f"r_{flat:05d}.png"
        if not svp.exists() or not (D3/f"{flat:05d}.png").exists(): continue
        c2w=np.asarray(f["transform_matrix"],float);M=np.linalg.inv(c2w)
        Rc=-np.transpose(M[:3,:3]);Rc[:,0]=-Rc[:,0];Tc=-M[:3,3]
        rA=render_one(A,g,xyz_t,Rc,Tc,t,fov_x,pipe,bg,FovY,N)
        rB=render_one(B,g,xyz_t,Rc,Tc,t,fov_x,pipe,bg,FovY,N)
        sv=loadimg(svp);d3=loadimg(D3/f"{flat:05d}.png")
        rv=float(np.abs(rA-sv).mean()); rc=float(np.abs(rB-d3).mean()); rw=float(np.abs(sv-d3).mean())
        for st_,k in ((baz,v2az[v]),(bel,v2el[v])):
            st_[k]["v"].append(rv);st_[k]["c"].append(rc);st_[k]["raw"].append(rw)
        gmap=np.abs(rA-sv).mean(-1)-np.abs(rB-d3).mean(-1)
        pix[v]["g"]+=gmap; pix[v]["n"]+=1
        done+=1
        if done%150==0: print(f"  {done} done",flush=True)
    def agg(store):
        ks=sorted(store)
        Rv=np.array([np.mean(store[k]["v"]) for k in ks]);Rc=np.array([np.mean(store[k]["c"]) for k in ks])
        raw=np.array([np.mean(store[k]["raw"]) for k in ks]);return np.array(ks),Rv,Rc,raw
    from scipy.stats import spearmanr,pearsonr
    azk,aRv,aRc,araw=agg(baz); gap=aRv-aRc; pr,_=pearsonr(gap,araw); rho,_=spearmanr(gap,araw)
    elk,eRv,eRc,eraw=agg(bel); egap=eRv-eRc; epr,_=pearsonr(egap,eraw)
    print(f"\n[{a.tag}] AZIMUTH:")
    print(f"{'az':>5} {'R_vgm':>8} {'R_clean':>8} {'gap':>8} {'raw':>8}")
    for k,rv,rc,g_,r in zip(azk,aRv,aRc,gap,araw): print(f"{k:>5.0f} {rv:>8.4f} {rc:>8.4f} {g_:>8.4f} {r:>8.4f}")
    print(f"R_clean range(az)={aRc.max()-aRc.min():.4f}")
    print(f"corr(gap,raw) az: Pearson={pr:.3f} Spearman={rho:.3f}")
    print(f"corr(gap,raw) elev: Pearson={epr:.3f}")
    np.savez(REPO/f"runs_aux/fit_residual_{a.tag}.npz",azk=azk,aRv=aRv,aRc=aRc,araw=araw,
             elk=elk,eRv=eRv,eRc=eRc,eraw=eraw)
    # per-pixel map for worst-azimuth view
    worst_az=azk[int(gap.argmax())]; wv=[v for v in pix if v2az[v]==worst_az][0]
    gm=pix[wv]["g"]/max(pix[wv]["n"],1)
    np.save(REPO/f"runs_aux/fit_residual_{a.tag}_pixmap.npy",gm)
    print(f"saved npz + pixmap (worst az={worst_az:.0f}, view {wv})")
    print("VERDICT:", "REPRODUCES cone" if pr>0.6 else "does NOT reproduce (absorption/flat)")
if __name__=="__main__": main()
