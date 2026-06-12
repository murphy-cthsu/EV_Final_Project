"""SED on SELF-GENERATED sv4d2 5-view multi-ring output (GT-free, known analytic cameras).
Layout: <root>/<scene>_5v_elev{E}/sv4d2/000000_v00{1..4}.mp4, v-index -> azimuth [60,120,180,240].
Poses: reuse lego_v3 transforms_sv4d2_math poses by (elev,az) tag — relative orbit geometry is
scene-independent. Same-azimuth adjacent-elevation pairs; symmetric epipolar distance.
"""
import json, argparse
from pathlib import Path
import numpy as np
import cv2, imageio
from collections import defaultdict
REPO=Path("/home/cthsu/EV_Final_Project"); T=21; H=W=576
AZ_OF_V={1:60,2:120,3:180,4:240}
def w2c_cv(c2w):
    return np.diag([1.,-1.,-1.,1.])@np.linalg.inv(c2w)
def fundamental(c2w_a,c2w_b,K):
    Rel=w2c_cv(c2w_b)@np.linalg.inv(w2c_cv(c2w_a)); R,t=Rel[:3,:3],Rel[:3,3]
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    Ki=np.linalg.inv(K); return Ki.T@(tx@R)@Ki
def sed_pair(ga,gb,F,sift,bf):
    ka,da=sift.detectAndCompute(ga,None); kb,db=sift.detectAndCompute(gb,None)
    if da is None or db is None or len(ka)<10 or len(kb)<10: return None
    m=bf.knnMatch(da,db,k=2)
    good=[p for p,q in (x for x in m if len(x)==2) if p.distance<0.75*q.distance]
    if len(good)<15: return None
    pa=np.float32([ka[g.queryIdx].pt for g in good]); pb=np.float32([kb[g.trainIdx].pt for g in good])
    pah=np.hstack([pa,np.ones((len(pa),1))]); pbh=np.hstack([pb,np.ones((len(pb),1))])
    la=(F@pah.T).T; lb=(F.T@pbh.T).T
    d1=np.abs((pbh*la).sum(1))/np.sqrt(la[:,0]**2+la[:,1]**2)
    d2=np.abs((pah*lb).sum(1))/np.sqrt(lb[:,0]**2+lb[:,1]**2)
    return float(np.median(0.5*(d1+d2)))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/mnt/HDD_1/cthsu/sv4d_p1_out")
    ap.add_argument("--scene",required=True); ap.add_argument("--elevs",default="0,5,10")
    a=ap.parse_args()
    elevs=[int(x) for x in a.elevs.split(",")]
    # pose lookup from lego_v3 math transforms (orbit geometry scene-independent)
    meta=json.loads((REPO/"data/custom/lego_v3/transforms_train.json").read_text())
    frames=meta["frames"]+json.loads((REPO/"data/custom/lego_v3/transforms_test.json").read_text())["frames"]
    fov_x=meta["camera_angle_x"]; fx=(W/2)/np.tan(fov_x/2)
    K=np.array([[fx,0,W/2],[0,fx,H/2],[0,0,1]])
    pose={}
    for f in frames:
        pose[(int(f["elevation_deg"]),int(f["azimuth_deg"]))]=np.asarray(f["transform_matrix"],float)
    def vid(scene,e,vi):
        p=Path(a.root)/f"{scene}_5v_elev{e}"/"sv4d2"/f"000000_v{vi:03d}.mp4"
        if not p.exists(): return None
        r=imageio.get_reader(str(p))
        return [cv2.cvtColor(r.get_data(t),cv2.COLOR_RGB2GRAY) for t in range(0,T,5)]
    sift=cv2.SIFT_create(); bf=cv2.BFMatcher()
    res=defaultdict(list)
    for vi,az in AZ_OF_V.items():
        vids={e:vid(a.scene,e,vi) for e in elevs}
        for e1,e2 in zip(elevs[:-1],elevs[1:]):
            if vids[e1] is None or vids[e2] is None or (e1,az) not in pose or (e2,az) not in pose: continue
            F=fundamental(pose[(e1,az)],pose[(e2,az)],K)
            for ga,gb in zip(vids[e1],vids[e2]):
                r=sed_pair(ga,gb,F,sift,bf)
                if r is not None: res[az].append(r)
    azs=sorted(res); vals=np.array([np.median(res[z]) for z in azs])
    print(f"[{a.scene} self-gen] SED by azimuth (px, median):")
    for z,v in zip(azs,vals): print(f"  az{z:>4}: {v:6.2f}  (n={len(res[z])})")
    print(f"  overall median: {np.median(np.concatenate([res[z] for z in azs])):.2f} px")
    np.savez(REPO/f"runs_aux/sed_selfgen_{a.scene}.npz",azs=azs,vals=vals)
if __name__=="__main__": main()
