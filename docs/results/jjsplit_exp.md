# jumpingjacks_splitting — our method(all-move),self-supervised

> 2026-06-15 · 來源 bundle:`jumpingjacks_splitting/`(結構同 shake_lefthand:原始 jumpingjacks 的 SC-GS,mlp deform @ iter40000 + 17 視角 render mp4 監督 + `transforms_sv4d2_math.json` 軌道姿態)。
> **無 clean reference**(正常);監督 = bundle 自己的 17 視角 mp4(灰底 dist-from-bg 抽 alpha)。

---

## 動作:做「劈腿 / splits」

從站姿(t=0)雙腿逐漸張開,到 t=20 完全側劈。**大幅度腿部關節動作**,所以這次直接跑 **all-move**(全部 Gaussian 都可動)。

GT t-strip(t=0→20):站立 → 漸開 → 全劈。

---

## 設定

| 項目 | 值 |
|---|---|
| canonical | 乾淨 `scene_point_cloud.ply`(74244 G,站姿,與 splitting t=0 吻合) |
| 動作 | **all-move**:`arm_weights` 全 1,K=100 cluster SE(3)+LBS+per-time scale+xyz residual,`--zero_traj_init` |
| 監督 | 17 視角 × 21 幀 mp4(self-supervised;v5_render_dir=self-ref ⇒ 純 L1 photo) |
| 訓練 | 8000 iters |

## 結果

| 設定 | held-out PSNR |
|---|---:|
| all-move(clean canonical) | **17.12 dB** |

**視覺對照**(每列一個 t,左 GT splitting | 右 ours all-move):

![](../../runs_aux/jjsplit_viz/keyframes.png)

訓練視角動畫(view0):

![](../../runs_aux/jjsplit_viz/train_view0.gif)

**4D novel-view 環繞**(相機繞圈 + 時間推進):

![](../../runs_aux/jjsplit_viz/novel4d.gif)

---

## 觀察(誠實)

- **軀幹/頭/手臂重建尚可**,all-move **抓到劈腿的大致動作**(腿確實隨 t 張開)。
- **極端劈腿(t=10–20)有明顯拖影 + 橘色 floaters 碎裂**:從**站姿** canonical 把腿拉到**全側劈**是很大的形變,cluster SE(3)+xyz residual 在這種大關節角度下會噴出雜點/streaking。
- 這也凸顯 all-move 的弱點:**無結構先驗、所有 Gaussian 自由動 → 大形變時容易碎**。若有對應「劈腿中段」姿勢的乾淨 canonical,或加 ARAP/部位約束,極端姿勢會更穩。

## 重現

```bash
PY=/home/cthsu/miniconda3/envs/scgs/bin/python
# 1. 監督:bundle 的 17 mp4 + sv4d2_math 姿態
$PY scripts/build_bundle_from_mp4.py --bundle jumpingjacks_splitting \
    --viddir jumpingjacks_splitting_r10_train --scene jumpingjacks_splitting
# 2. all-move part_dir：arm_weights 全 1（N=74244，對齊 scene_point_cloud.ply）
# 3. Stage E all-move
CANON=outputs/custom/jjshake_cleanjj_canon/point_cloud/iteration_0/point_cloud.ply  # = scene_point_cloud.ply
$PY scripts/train_partrigid_hier.py --label jjsplit_allmove --canon_ply $CANON \
    --part_dir runs_aux/part_assignment_jumpingjacks_splitting_allmove \
    --scene_dir data/custom/jumpingjacks_splitting --v5_render_dir outputs/custom/jumpingjacks_splitting_selfref/renders \
    --k_arm 100 --lbs_K 6 --lam_arap 1.0 --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --zero_traj_init --iterations 8000
```
