"""Unified 'reliability cone' summary figure tying D5 (spatial), D3 (temporal),
D6 (pose) into one narrative. Binned values are from diagnose_vgm_inconsistency.py
and diagnose_vgm_temporal_flicker.py (lego_v3, SV4D 2.0 vs clean d-3dgs)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/home/cthsu/EV_Final_Project/meetTW_checkpoint_0601/figs/vgm_reliability_cone.png")

# --- binned results (source: the two diagnose_* scripts) ---
az      = np.array([0,60,120,150,180,210,240,270,330], float)
psnr_az = np.array([37.5,23.3,19.4,19.7,24.2,22.9,21.8,22.8,27.5])      # D5 spatial
flick_az= np.array([0.0092,0.0430,0.0234,0.0257,0.0292,0.0284,0.0383,0.0210,0.0383])  # D3 temporal
el      = np.array([0,5,10,15,20,25,30], float)
psnr_el = np.array([23.9,22.8,21.8,26.3,23.6,21.2,20.8])                 # D5 spatial
drift_el= np.array([46.5,54.6,55.8,55.6,62.0,63.8,64.8])                 # D6 pose drift px

def norm01(x): return (x-x.min())/(x.max()-x.min()+1e-9)
# reliability (higher=better): spatial=PSNR up; temporal=low flicker -> 1-norm(flicker)
rel_spatial = norm01(psnr_az)
rel_temporal= 1-norm01(flick_az)

fig = plt.figure(figsize=(15,5))

# Panel 1: the reliability cone (azimuth) — spatial + temporal overlaid
ax1 = fig.add_subplot(1,3,1,projection="polar")
a = np.deg2rad(np.append(az,az[0]))
for vals,c,lab in [(np.append(rel_spatial,rel_spatial[0]),"#b85450","spatial fidelity (D5)"),
                   (np.append(rel_temporal,rel_temporal[0]),"#d6840b","temporal stability (D3)")]:
    ax1.plot(a,vals,"o-",color=c,lw=2,ms=5,label=lab); ax1.fill(a,vals,color=c,alpha=.10)
ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
ax1.set_rticks([0,.5,1]); ax1.set_title("Reliability cone vs azimuth\n(input view at 0°; outer = more reliable)",pad=22,fontsize=11)
ax1.legend(loc="upper right",bbox_to_anchor=(1.25,1.12),fontsize=8)

# Panel 2: spatial fidelity raw dB (azimuth) — the absolute numbers
ax2 = fig.add_subplot(1,3,2)
order = np.argsort(az)
ax2.bar(np.arange(len(az)), psnr_az[order], color="#b85450", alpha=.85)
ax2.set_xticks(np.arange(len(az))); ax2.set_xticklabels([f"{int(x)}°" for x in az[order]],fontsize=8)
ax2.axhline(psnr_az.max(),ls="--",color="gray",alpha=.5)
ax2.set_ylabel("PSNR SV4D vs clean GT (dB)"); ax2.set_xlabel("azimuth from input view")
ax2.set_title(f"Spatial fidelity: {psnr_az.max():.0f} dB at input → {psnr_az.min():.0f} dB far side",fontsize=11)
ax2.grid(axis="y",alpha=.3)

# Panel 3: elevation degradation — spatial (dB) + pose drift (px) twin axis
ax3 = fig.add_subplot(1,3,3)
s,i = np.polyfit(el,psnr_el,1)
l1=ax3.plot(el,psnr_el,"o-",color="#b85450",lw=2,ms=6,label=f"spatial (D5) {s*10:.2f} dB/10°")
ax3.plot(el,s*el+i,"--",color="#b85450",alpha=.4)
ax3.set_xlabel("elevation from input view (deg)"); ax3.set_ylabel("PSNR (dB)",color="#b85450")
ax3.tick_params(axis="y",labelcolor="#b85450")
ax3b=ax3.twinx()
so,io=np.polyfit(el,drift_el,1)
l2=ax3b.plot(el,drift_el,"s-",color="#82b366",lw=2,ms=6,label=f"pose drift (D6) +{so:.1f} px/°")
ax3b.set_ylabel("centroid drift (px)",color="#82b366"); ax3b.tick_params(axis="y",labelcolor="#82b366")
ax3.set_title("Degradation with elevation",fontsize=11)
ls=l1+l2; ax3.legend(ls,[x.get_label() for x in ls],fontsize=8,loc="center right")
ax3.grid(alpha=.3)

fig.suptitle("SV4D 2.0 is reliable in a cone around the input view — and degrades spatially, temporally, and in pose off-axis",
             fontsize=13,fontweight="bold",y=1.02)
plt.tight_layout()
plt.savefig(OUT,dpi=120,bbox_inches="tight")
print("saved",OUT)
