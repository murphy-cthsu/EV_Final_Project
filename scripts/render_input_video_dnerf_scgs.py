"""Render a fixed-camera 21-frame input video from a trained SC-GS deformable model
(mirrors upstream's 'lego_r7_train' input). For SV4D conditioning on new scenes."""
import json, sys, argparse
from pathlib import Path
import numpy as np
import torch
import imageio
REPO=Path("/home/cthsu/EV_Final_Project"); sys.path.insert(0,str(REPO/"third_party"/"SC-GS"))
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from scene.cameras import Camera as SCGSCamera
from gaussian_renderer import render
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal
H=W=576
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scene",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--n_frames",type=int,default=21)
    a=ap.parse_args()
    mp=Path(a.model)
    it=max(int(p.name.split("_")[-1]) for p in (mp/"point_cloud").iterdir() if p.name.startswith("iter"))
    ds=torch.load(mp/"deform"/f"iteration_{it}"/"deform.pth",map_location="cuda")
    node_num=ds["nodes"].shape[0]; hyper_dim=ds["nodes"].shape[1]-3
    g=GaussianModel(3,fea_dim=8,with_motion_mask=False)
    g.load_ply(str(mp/"point_cloud"/f"iteration_{it}"/"point_cloud.ply"),og_number_points=0)
    deform=DeformModel(K=4,deform_type="node",is_blender=True,skinning=False,hyper_dim=hyper_dim,
                       node_num=node_num,pred_opacity=False,pred_color=False,use_hash=False,
                       hash_time=False,d_rot_as_res=True,local_frame=True,
                       progressive_brand_time=False,with_arap_loss=True,max_d_scale=-1,
                       enable_densify_prune=False,is_scene_static=False)
    deform.load_weights(str(mp),iteration=it)
    meta=json.loads((REPO/f"data/dnerf/{a.scene}/transforms_test.json").read_text())
    fov_x=meta["camera_angle_x"]; FovY=focal2fov(fov2focal(fov_x,W),H)
    c2w=np.asarray(meta["frames"][0]["transform_matrix"],float)   # fixed camera = test frame 0
    M=np.linalg.inv(c2w); R=-np.transpose(M[:3,:3]); R[:,0]=-R[:,0]; Tc=-M[:3,3]
    pa=_A(); pipe=PipelineParams(pa).extract(pa.parse_args([]))
    bg=torch.tensor([1.,1,1],device="cuda")
    frames=[]
    for t in range(a.n_frames):
        fid=torch.tensor(t/max(a.n_frames-1,1)).float()
        cam=SCGSCamera(colmap_id=0,R=R,T=Tc,FoVx=fov_x,FoVy=FovY,image=torch.zeros(3,H,W).cuda(),
                       gt_alpha_mask=None,image_name="x",uid=0,fid=fid)
        ti=deform.deform.expand_time(cam.fid.to("cuda"))
        with torch.no_grad():
            d=deform.step(g.get_xyz.detach(),ti,feature=g.feature,
                          motion_mask=getattr(g,"motion_mask",None),is_training=False)
            pkg=render(cam,g,pipe,bg,d_xyz=d["d_xyz"],d_rotation=d["d_rotation"],
                       d_scaling=d["d_scaling"],d_opacity=d.get("d_opacity"),
                       d_color=d.get("d_color"),d_rot_as_res=deform.d_rot_as_res)
        img=(torch.clamp(pkg["render"],0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        frames.append(img)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    imageio.mimwrite(a.out,frames,fps=10,quality=9)
    print(f"[{a.scene}] wrote {a.out} ({len(frames)} frames, iter {it})")
if __name__=="__main__": main()
