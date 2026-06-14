import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
REPO=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPO/"scripts")); sys.path.insert(0,str(REPO/"third_party"/"SC-GS"))
import torch
from arguments import PipelineParams
from argparse import ArgumentParser as _A
from utils.graphics_utils import focal2fov, fov2focal
import eval_region_psnr as E
VIEWS={"az0":"elev_0_az_0","az270":"elev_0_az_270"}; T=10
m=json.loads((REPO/"data/custom/lego_v3/transforms_train.json").read_text())
mt=json.loads((REPO/"data/custom/lego_v3/transforms_test.json").read_text())
allf=m["frames"]+mt["frames"]; fov_x=m["camera_angle_x"]; FovY=focal2fov(fov2focal(fov_x,576),576)
by={(f["view_tag"],int(f["frame_idx"])):f for f in allf}
_pq=_A(); pp=PipelineParams(_pq); pipe=pp.extract(_pq.parse_args([]))
bg=torch.tensor([1,1,1],dtype=torch.float32,device="cuda")
ours=E.make_partrigid_renderer("lego_v2_A1_leakfree",pipe,bg,d_rot_zero=True)
out=REPO/"runs_aux/recon_inout"; out.mkdir(parents=True,exist_ok=True)
for k,tag in VIEWS.items():
    f=by[(tag,T)]; img=ours(f,fov_x,FovY,21)
    Image.fromarray((img*255).astype(np.uint8)).save(out/f"ours_{tag}.png"); print("saved ours",tag)
