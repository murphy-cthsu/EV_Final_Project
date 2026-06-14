# Vanilla SC-GS(原版,無任何 prior canonical)— lego_v2 & standup

> 2026-06-14 · 重跑 SC-GS 原始版本作為 baseline:**隨機初始化 + joint 訓練(結構+motion,16M deform-MLP),不給任何 prior canonical**。
> 訓練 20k iters · node deform(node_num=512, hyper_dim=8)· eval vs SV4D 監督資料 **與** 獨立 clean d-3dgs GT。
> 模型:`outputs/custom/{lego_v2,standup}_vanilla_noprior_node/` · log:`runs_aux/vanilla_noprior_logs/`

---

## 數字

| Scene | 初始化 | N (Gaussians) | vs SV4D | **vs clean d-3dgs GT** | gap (d3dgs−sv4d) | 文件記載 |
|---|---|---:|---:|---:|---:|---:|
| **lego_v2** | random 100k(canonical 移開) | 62k | 12.81 | **11.38 dB** | −1.43 | 11.43 ✓ |
| **standup** | random 100k | 188k | 6.51 | **6.60 dB** | +0.09 | 6.63 ✓ |

兩者都重現了文件裡的 vanilla baseline(差 <0.05 dB)。`gap` 為負/接近 0 → 模型擬合的是**噪聲 SV4D**、而非乾淨 GT,VGM 幻覺被吸進幾何 → 教科書級放射狀黑針爆炸。

---

## Failure grid(海報用,2×3)

左上 = clean d-3dgs GT;其餘 5 格 = vanilla SC-GS 在 t=10 從不同方位角(0/72/144/216/288°)render。**每個角度都爆放射狀黑針** → 原版 SC-GS 在 VGM(SV4D)監督下幾何全毀。

![](../../runs_aux/vanilla_noprior_viz/lego_v2/vanilla_failure_grid.png)

standup(左上 = clean GT 人物,其餘 5 角度 vanilla render 被黑針團包覆):

![](../../runs_aux/vanilla_noprior_viz/standup/vanilla_failure_grid.png)

---

## lego_v2

**Keyframes**(每列一個時間 t,欄位:SV4D 監督 | clean d-3dgs GT | vanilla SC-GS):

![](../../runs_aux/vanilla_noprior_viz/lego_v2/keyframes.png)

**訓練視角動畫**(view 0,21 幀,SV4D | clean GT | vanilla 三欄):

![](../../runs_aux/vanilla_noprior_viz/lego_v2/train_view0.gif)

**Novel-view turntable**(vanilla only,環繞 48 幀 + 時間推進,訓練未見視角):

![](../../runs_aux/vanilla_noprior_viz/lego_v2/novel_orbit.gif)

→ vanilla 欄整顆 Gaussian 在 silhouette 邊界爆出放射狀黑針;clean GT 欄是乾淨的怪手。novel view 下爆炸更明顯(沒有監督視角撐著)。

---

## standup

**Keyframes**:

![](../../runs_aux/vanilla_noprior_viz/standup/keyframes.png)

**訓練視角動畫**(view 0,SV4D | clean GT | vanilla):

![](../../runs_aux/vanilla_noprior_viz/standup/train_view0.gif)

**Novel-view turntable**(vanilla only):

![](../../runs_aux/vanilla_noprior_viz/standup/novel_orbit.gif)

→ 人物被黑針團包覆,6.6 dB 是四個場景裡最低 — standup 動態大 + SV4D 監督不一致最嚴重。

---

## 重現指令

```bash
PY=/home/cthsu/miniconda3/envs/scgs/bin/python

# 確保「無 prior canonical」:lego_v2 資料夾原本的 points3d.ply 其實是 114k canonical,
# 訓練前移開讓 SC-GS 自動生成隨機 100k 點(standup 本來就是隨機,不用動)。
mv data/custom/lego_v2/points3d.ply data/custom/lego_v2/points3d.ply.canon_bak   # 跑完記得還原

for S in lego_v2 standup; do
  CUDA_VISIBLE_DEVICES=0 $PY third_party/SC-GS/train_gui.py \
      --source_path data/custom/$S \
      --model_path outputs/custom/${S}_vanilla_noprior \
      --deform_type node --node_num 512 --hyper_dim 8 \
      --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
      --resolution 1 --W 576 --H 576 --iterations 20000
done

# eval(vs SV4D + vs clean d-3dgs GT)
$PY scripts/eval_vanilla_lego_v2.py --model_path outputs/custom/lego_v2_vanilla_noprior_node
$PY scripts/eval_vanilla_lego_v2.py --model_path outputs/custom/standup_vanilla_noprior_node \
    --scene_dir data/custom/standup --d3dgs_dir outputs/custom/standup_d3dgs_ref/renders

# 視覺化(render / novel view / animation)
$PY scripts/viz_vanilla_noprior.py --model outputs/custom/lego_v2_vanilla_noprior_node \
    --scene_dir data/custom/lego_v2 --d3dgs_dir outputs/custom/lego_v2_d3dgs_ref/renders \
    --scene lego_v2 --orbit_elev -15
$PY scripts/viz_vanilla_noprior.py --model outputs/custom/standup_vanilla_noprior_node \
    --scene_dir data/custom/standup --d3dgs_dir outputs/custom/standup_d3dgs_ref/renders \
    --scene standup --orbit_elev -25

mv data/custom/lego_v2/points3d.ply.canon_bak data/custom/lego_v2/points3d.ply   # 還原 canonical
```
