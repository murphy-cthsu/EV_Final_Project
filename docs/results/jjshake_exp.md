# jumpingjacks_shake_lefthand — part-rigid 重建(self-supervised,無 clean reference)

> 2026-06-15 · 來源 bundle:`jumpingjacks_shake_lefthand.zip`。
> **修正版**:第一次我用錯監督(自己重新 render deform 的完整 fid 週期 → 跑出全身 jumping jacks);本頁是用 **bundle 提供的 mp4(只有左手抖)+ 其估計軌道姿態**的正確結果。

---

## ⚠️ 第一次的錯誤(為什麼曾經「學出腳會開合」)

bundle 的 deform 是**原始完整 jumpingjacks** 模型。我一開始 `build_jjshake_selfgen.py` 把它用 `fid=t/20` 掃完整時間軸自己 render → 得到全身跳躍(腿開合),還把視角 elev 用成 0(實際軌道 15°)。那不是 bundle 要的內容。

| | 內容 |
|---|---|
| **provided mp4(正確監督)** | 全程舉手站姿,**只有左手抖**(殘影),腿幾乎不動 |
| 我第一次餵的 re-render(錯) | arms up/down + 腿開合的完整 jumping jacks |

修正:`build_jjshake_from_mp4.py` 改用 bundle 的 17 支 mp4 + `camera_estimation_math/transforms_sv4d2_math.json` 的軌道姿態(半徑 4.03、elev 15°),灰底用 dist-from-bg 抽 alpha。重跑 Stage B/C/D 後 motion = **11.9%**,與 bundle 提供的 tuned 14% 吻合 → 證實腿確實不動、provided 分類本來就是對的。

---

## 結果總表(正確監督,held-out test)

| canonical | 動作設定 | held-out PSNR | 視覺 |
|---|---|---:|---|
| bundle 模糊 canonical | 12% 凍結身體 | 15.03 dB | 模糊爆炸,身體對不上 |
| 乾淨 canonical(自訓單姿勢+prune) | 12% shake-only | 16.60 dB | 身體乾淨,手部小瑕疵 |
| 乾淨 canonical(自訓) | 100% all-move | 17.15 dB | PSNR 略高但黑針 |
| **乾淨 canonical(從 clean jj 訓,`scene_point_cloud.ply`)** | shake-only,auto mask 18% | **17.34 dB** | **身體最乾淨**,姿勢吻合 |
| 同上 | shake-only,tuned mask 16.6%(左臂) | 17.02 dB | 只左臂當動,語意最對但 PSNR 略低 |
| 同上 | 100% all-move | **18.08 dB** | PSNR 最高但身體雜訊/拖影較多 |

三個重點:
1. **canonical 品質直接決定上限**:模糊 bundle canonical 15.0 → 自訓單姿勢 16.6 → 從 clean jj 訓的 `scene_point_cloud.ply` **17.3 dB**(最佳;這顆乾淨且姿勢天生吻合舉手監督,連 prune 都不用)。
2. 每一顆乾淨 canonical 上,**all-move 的 PSNR 都比 shake-only 高 ~0.5–0.7 dB,但視覺更差**(身體被注入假動作、長拖影)。PSNR 偏好過擬合邊緣雜訊,**shake-only 結構上才正確** → 「凍結 canonical + 只學該動的部分 > 讓全部都動」的證據。
3. 所以本場景的正解 = **乾淨 canonical + shake-only**(身體凍結、只學手)。

### 為什麼第一版(模糊 canonical)只有 15 dB
bundle 的 canonical 是 **Deformable-3DGS 的 canonical 那組 3DGS**(它本身就是一組 3D Gaussians,不是 MLP;但它是跟 deform MLP 一起訓出的「整段動作平均態」)。直接 render(不變形)是模糊、手臂朝下的 rest pose,不是 lego 那種對單一乾淨幀訓的 frozen 3DGS。凍結它 → 佔 88% 的靜態身體卡在模糊姿勢,對不上乾淨舉手監督。

> 註:bundle 的 SC-GS 模型「canonical + MLP-deform 一起」render 是正常的(那些 mp4 就是證據);問題是**單獨把它的 canonical 拿來凍結**會退化。

### canonical(def=0)vs 監督

![](../../runs_aux/jjshake_viz/canon_vs_sup.png)

左 = 凍結基準(模糊、手臂朝下);右 = 監督(乾淨舉手、只手抖)。

---

## 視覺對照(正確監督)

**Keyframes**(view0;左 GT 只手抖 | 右 ours 凍結模糊 canonical):

![](../../runs_aux/jjshake_viz/v2_keyframes.png)

**訓練視角動畫**(view0,GT | ours):

![](../../runs_aux/jjshake_viz/v2_train_view0.gif)

---

## Bundle 提供的 motion 分類(Stage B/C)

每個 Gaussian 的 motion/static 正交視圖(紅=motion,集中在手臂/抖手):

![](../../outputs/custom/jumpingjacks_shake_lefthand/motion_classification/gaussians_orthographic.png)

單視角 motion overlay(紅 = 左手抖動拖影):

![](../../outputs/custom/jumpingjacks_shake_lefthand/motion_classification/motion_masks/elev_0_az_0/motion_overlay_midframe.png)

---

## ⭐ 最佳:從 clean jumpingjacks 訓的 canonical(`scene_point_cloud.ply`)

使用者另外用**原始乾淨 jumpingjacks** 訓了一顆 3DGS canonical(74244 Gaussians)。它**乾淨、無黑針,且姿勢天生吻合**舉手監督 —— 直接可用,不需 prune。

**canonical(def=0)vs 監督**(左=監督舉手,右=canonical,3 視角):

![](../../runs_aux/jjshake_viz/cleanjj_canon_vs_sup.png)

**shake-only(凍結身體,18% 手)vs all-move(100%)**,每列一個 t(GT | shake-only | all-move):

![](../../runs_aux/jjshake_viz/cleanjj_keyframes.png)

訓練視角動畫(view0):

![](../../runs_aux/jjshake_viz/cleanjj_train_view0.gif)

→ shake-only **17.34 dB**、all-move **18.08 dB**。all-move PSNR 高但身體雜訊較多;shake-only 身體最乾淨且結構正確。**這是目前最佳的一顆 canonical**(part_dir `_cleanjj`,labels `partrigid_jjshake_cleanjj_{shakeonly,allmove}`)。

### 新 tuned motion mask 實驗(`motion_classification/`,對齊這顆 canonical)

使用者另提供對齊這顆 canonical 的 tuned 分類(`motion_classification/gaussian_labels.npy`,74244 對齊,**16.6% motion,集中在左臂/手**;otsu×0.9 + blur/morph 7)。直接注入 `arm_weights` 跑 shake-only:

| mask | motion | held-out PSNR | 視覺 |
|---|---:|---:|---|
| auto(motion_parts_generic) | 18.3% | **17.34 dB** | 兩臂都當動,手臂 fuzz |
| **tuned(`motion_classification/`)** | 16.6%(左臂) | 17.02 dB | 只左臂當動、右臂凍結,更貼「只左手抖」語意 |

![](../../runs_aux/jjshake_viz/tunedmask_keyframes.png)

(GT | tuned mask 16.6% | auto mask 18.3%)

**誠實結論**:tuned mask 更精準(只標真正在抖的左臂),但 PSNR **略低 0.3 dB** —— 因為它把右臂也凍住,而監督裡整個人有輕微晃動,凍右臂賠一點 PSNR。**與全頁主軸一致:更精準/更少的 motion 集合 ≠ 更高 PSNR(PSNR 偏好放更多自由度去過擬合)。** 本場景瓶頸是手臂自監督重建的 fuzz,不是 mask 好壞;tuned mask 在語意/結構上才是對的拆解。

tuned-mask 的 4D novel-view 環繞:

![](../../runs_aux/jjshake_viz/novel4d_tunedmask.gif)

### Novel-view 4D 環繞(相機繞一圈 + 時間同步推進,訓練未見視角)

**shake-only**(凍結身體、只學手):

![](../../runs_aux/jjshake_viz/novel4d_shakeonly.gif)

**all-move**(全動,對照):

![](../../runs_aux/jjshake_viz/novel4d_allmove.gif)

(軌道用訓練 view0 的 c2w 繞世界 Z 軸旋轉生成,保證與訓練世界座標一致;72 幀,az 0→360 同時 t 0→20。)

---

## Clean-canonical 路線(我自己對單一乾淨幀訓 3DGS,較早嘗試)+ all-move 對照

把 17 視角在單一時刻(t=10)的 shake 影像當「單姿勢多視角」訓一顆靜態 3DGS(`jjshake_canon_src` → `train_gui.py`),姿勢天生與監督一致。原始輸出有 vanilla 式黑針爆炸(N densify 到 91k),用 visual-hull + scale + opacity prune 清成 48959 個乾淨 Gaussians。

**raw(爆炸)→ pruned(乾淨)vs 監督**:

![](../../runs_aux/jjshake_viz/clean_canon_vs_sup.png)

(左=監督,右=pruned 乾淨 canonical,def=0 直接 render,姿勢吻合)

### shake-only(12%)vs all-move(100%),同一顆乾淨 canonical

每列一個 t;欄位:**GT 只手抖 | shake-only 凍結身體 | all-move 全動**:

![](../../runs_aux/jjshake_viz/clean_keyframes.png)

訓練視角動畫(view0):

![](../../runs_aux/jjshake_viz/clean_train_view0.gif)

→ **shake-only 身體乾淨**(凍結=正確,身體本就不動);**all-move 身體長黑針/拖影**,PSNR 雖高 0.5 dB 是靠過擬合邊緣雜訊。**這正是 freeze-prior 的價值**:只讓該動的(手)動,結構先驗保住靜態身體。

> all-move **不需要**自己的 canonical;它用同一顆乾淨 canonical(`arm_weights` 全 1 + `--zero_traj_init`),才能與 shake-only 公平對照。

---

## 接下來可選

- shake-only 的身體其實有輕微「整體晃動」(人物會 bob),加一個 global rigid 補償可再拉高 PSNR。
- all-move 黑針 close-up 放大圖。

---

## 重現

```bash
PY=/home/cthsu/miniconda3/envs/scgs/bin/python
unzip -oq jumpingjacks_shake_lefthand.zip -d outputs/custom/

# 正確監督:provided mp4 + sv4d2_math 軌道姿態
$PY scripts/build_jjshake_from_mp4.py

# Stage B/C/D（用 provided mp4，motion≈12%）
$PY scripts/motion_parts_generic.py --dataset jumpingjacks_shake_lefthand \
    --canon_ply outputs/custom/jumpingjacks_shake_lefthand/point_cloud/iteration_40000/point_cloud.ply \
    --src_video_root outputs/custom/jumpingjacks_shake_lefthand/jumpingjacks_shake_leftthand_r10_train_iter40000 \
    --threshold_method otsu

# Stage E（self-supervised；v5_render_dir=self-ref ⇒ 純 L1 photo）
$PY scripts/train_partrigid_hier.py --label jjshake_v2 \
    --canon_ply outputs/custom/jumpingjacks_shake_lefthand/point_cloud/iteration_40000/point_cloud.ply \
    --part_dir runs_aux/part_assignment_jumpingjacks_shake_lefthand \
    --scene_dir data/custom/jumpingjacks_shake_lefthand \
    --v5_render_dir outputs/custom/jjshake_selfref/renders \
    --k_arm 100 --lbs_K 6 --lam_arap 1.0 --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --iterations 8000

# === Clean-canonical 路線 ===
# 1. 單姿勢多視角資料 (t=10, 17 views) → 訓靜態 3DGS canonical
#    (build script: 取 data/custom/jumpingjacks_shake_lefthand 的 t=10 各視角 → data/custom/jjshake_canon_src)
$PY third_party/SC-GS/train_gui.py --source_path data/custom/jjshake_canon_src \
    --model_path outputs/custom/jjshake_canonical --deform_type node --node_num 512 --hyper_dim 8 \
    --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 1 --W 576 --H 576 --iterations 15000
# 2. hull+scale+opacity prune 去黑針 → outputs/custom/jjshake_canonical_pruned/...
# 3. Stage B/C/D on clean canonical (--out_suffix _cleancanon)；motion≈12%
# 4. shake-only vs all-move（同一顆乾淨 canonical）
CANON=outputs/custom/jjshake_canonical_pruned/point_cloud/iteration_0/point_cloud.ply
$PY scripts/train_partrigid_hier.py --label jjshake_clean_shakeonly --canon_ply $CANON \
    --part_dir runs_aux/part_assignment_jumpingjacks_shake_lefthand_cleancanon \
    --scene_dir data/custom/jumpingjacks_shake_lefthand --v5_render_dir outputs/custom/jjshake_selfref/renders \
    --k_arm 100 --lbs_K 6 --lam_arap 1.0 --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --iterations 8000
# all-move: arm_weights 全 1 的 part_dir + --zero_traj_init
$PY scripts/train_partrigid_hier.py --label jjshake_clean_allmove --canon_ply $CANON \
    --part_dir runs_aux/part_assignment_jumpingjacks_shake_lefthand_allmove \
    --scene_dir data/custom/jumpingjacks_shake_lefthand --v5_render_dir outputs/custom/jjshake_selfref/renders \
    --k_arm 100 --lbs_K 6 --lam_arap 1.0 --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --zero_traj_init --iterations 8000

# === ⭐ 最佳:從 clean jj 訓的 canonical (scene_point_cloud.ply) ===
cp scene_point_cloud.ply outputs/custom/jjshake_cleanjj_canon/point_cloud/iteration_0/point_cloud.ply
CANON=outputs/custom/jjshake_cleanjj_canon/point_cloud/iteration_0/point_cloud.ply
$PY scripts/motion_parts_generic.py --dataset jumpingjacks_shake_lefthand --canon_ply $CANON \
    --src_video_root outputs/custom/jumpingjacks_shake_lefthand/jumpingjacks_shake_leftthand_r10_train_iter40000 \
    --threshold_method otsu --out_suffix _cleanjj   # motion≈18%
# shake-only
$PY scripts/train_partrigid_hier.py --label jjshake_cleanjj_shakeonly --canon_ply $CANON \
    --part_dir runs_aux/part_assignment_jumpingjacks_shake_lefthand_cleanjj \
    --scene_dir data/custom/jumpingjacks_shake_lefthand --v5_render_dir outputs/custom/jjshake_selfref/renders \
    --k_arm 100 --lbs_K 6 --lam_arap 1.0 --lam_photo_smart 3.0 --photo_smart_alpha 16.0 \
    --use_per_time_scale --use_xyz_residual --iterations 8000
# all-move: 另建 arm_weights 全 1 的 part_dir (_cleanjj_allmove) + --zero_traj_init
```
