"""SED (symmetric epipolar distance) — reference-free multi-view consistency, MEt3R's
geometric predecessor. Uses KNOWN cameras (no F estimation): if SV4D were 3D-consistent,
SIFT matches between two generated views must satisfy the epipolar constraint of the
REQUESTED cameras. Deviation = generator inconsistency (geometry/pose drift), no GT needed.
Pairs: same azimuth, adjacent elevations (small baseline -> SIFT reliable).
Control: same computation on clean d-3dgs (should be near-zero -> validates pipeline).
"""
import json, argparse
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

REPO=Path("/home/cthsu/EV_Final_Project"); T=21; H=W=576

def load_gray(p):
    im=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if im is None: return None
    if im.shape[-1]==4:
        a=im[...,3:4].astype(np.float32)/255
        rgb=im[...,:3].astype(np.float32)*a+255*(1-a)
        im=rgb.astype(np.uint8)
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

def w2c_cv(c2w):
    w2c=np.linalg.inv(c2w)
    return np.diag([1.,-1.,-1.,1.])@w2c   # blender->opencv

def fundamental(c2w_a,c2w_b,K):
    Ta,Tb=w2c_cv(c2w_a),w2c_cv(c2w_b)
    Rel=Tb@np.linalg.inv(Ta)              # cam_a -> cam_b
    R,t=Rel[:3,:3],Rel[:3,3]
    tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    E=tx@R
    Kinv=np.linalg.inv(K)
    return Kinv.T@E@Kinv

def sed_pair(ga,gb,F,sift,bf):
    ka,da=sift.detectAndCompute(ga,None); kb,db=sift.detectAndCompute(gb,None)
    if da is None or db is None or len(ka)<10 or len(kb)<10: return None
    m=bf.knnMatch(da,db,k=2)
    good=[p for p,q in (x for x in m if len(x)==2) if p.distance<0.75*q.distance]
    if len(good)<15: return None
    pa=np.float32([ka[g.queryIdx].pt for g in good])
    pb=np.float32([kb[g.trainIdx].pt for g in good])
    pa_h=np.hstack([pa,np.ones((len(pa),1))]); pb_h=np.hstack([pb,np.ones((len(pb),1))])
    la=(F@pa_h.T).T            # epipolar line in image b
    lb=(F.T@pb_h.T).T          # in image a
    d1=np.abs((pb_h*la).sum(1))/np.sqrt(la[:,0]**2+la[:,1]**2)
    d2=np.abs((pa_h*lb).sum(1))/np.sqrt(lb[:,0]**2+lb[:,1]**2)
    return float(np.median(0.5*(d1+d2))), len(good)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scene",required=True); a=ap.parse_args()
    SCENE=REPO/"data/custom"/a.scene; D3=REPO/"outputs/custom"/f"{a.scene}_d3dgs_ref"/"renders"
    meta=json.loads((SCENE/"transforms_train.json").read_text())
    frames=meta["frames"]+json.loads((SCENE/"transforms_test.json").read_text())["frames"]
    fov_x=meta["camera_angle_x"]
    fx=(W/2)/np.tan(fov_x/2); K=np.array([[fx,0,W/2],[0,fx,H/2],[0,0,1]])
    info={}
    for f in frames:
        v=f["view_idx"]
        if v not in info: info[v]=(f["elevation_deg"],f["azimuth_deg"],np.asarray(f["transform_matrix"],float))
    # same-azimuth adjacent-elevation pairs
    by_az=defaultdict(list)
    for v,(e,az,_) in info.items(): by_az[az].append((e,v))
    pairs=[]
    for az,lst in by_az.items():
        lst.sort()
        for i in range(len(lst)-1): pairs.append((az,lst[i][1],lst[i+1][1]))
    print(f"[{a.scene}] {len(pairs)} view pairs (same-az adjacent-elev)")
    sift=cv2.SIFT_create(); bf=cv2.BFMatcher()
    res=defaultdict(lambda:defaultdict(list))
    def img_path(side,v,t):
        flat=v*T+t
        if side=="sv":
            for sp in ("train","test"):
                c=SCENE/sp/f"r_{flat:05d}.png"
                if c.exists(): return c
            return None
        return D3/f"{flat:05d}.png"
    for az,va,vb in pairs:
        F=fundamental(info[va][2],info[vb][2],K)
        for t in range(0,T,5):
            for side in ("sv","d3"):
                pa,pb=img_path(side,va,t),img_path(side,vb,t)
                if pa is None or pb is None or not Path(pb).exists(): continue
                ga,gb=load_gray(pa),load_gray(pb)
                if ga is None or gb is None: continue
                r=sed_pair(ga,gb,F,sift,bf)
                if r: res[side][az].append(r[0])
    azs=sorted(set(res["sv"])&set(res["d3"]))
    sv=np.array([np.median(res["sv"][z]) for z in azs])
    d3=np.array([np.median(res["d3"][z]) for z in azs])
    print(f"\n{'az':>5} {'SED_SV4D(px)':>13} {'SED_clean(px)':>14} {'ratio':>7}")
    for z,s,d in zip(azs,sv,d3): print(f"{z:>5.0f} {s:>13.2f} {d:>14.2f} {s/max(d,1e-6):>7.1f}x")
    print(f"\nmedian: SV4D={np.median(sv):.2f}px clean={np.median(d3):.2f}px ratio={np.median(sv)/max(np.median(d3),1e-6):.1f}x")
    from scipy.stats import spearmanr
    # correlate with raw cone if available
    fr=REPO/f"runs_aux/fit_residual_{'legov3' if a.scene=='lego_v3' else a.scene}.npz"
    if fr.exists():
        d=np.load(fr); common=[i for i,z in enumerate(d["azk"]) if z in azs]
        if len(common)>=5:
            raw=d["araw"][common]; sed_m=np.array([sv[azs.index(d["azk"][i])] for i in common])
            rho,_=spearmanr(sed_m,raw)
            print(f"corr(SED, raw cone) Spearman={rho:.3f}")
    np.savez(REPO/f"runs_aux/sed_{a.scene}.npz",azs=azs,sv=sv,d3=d3)
if __name__=="__main__": main()
