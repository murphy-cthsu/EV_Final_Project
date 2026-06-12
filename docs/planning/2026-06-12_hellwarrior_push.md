# Hellwarrior push — Stage D 假說檢驗 + L2 多部位 + 3D-prompt SAM 原型（2026-06-12 夜）

> 動機:hellwarrior 重建差(13.5 dB vs lego 20.4)。候選假說:(a) motion mask 太 coarse
> /Stage D 單質心軌跡錯(open_issues I-1,標為「最可能主因」);(b) canonical pose
> 不對齊(蹲 vs 站,I-4 / handover §1c)。本 session 把 (a) 完整檢驗並排除。

## 1. 實驗階梯(全部同協議:leak-free motion-gated + rotfix `--d_rot_zero`,8000 iters,clean canonical)

| 實驗 | 改了什麼 | hellwarrior PSNR vs clean | Δ |
|---|---|---:|---:|
| control(`hellwarrior_cleancanon_ctrl_rotfix`) | 無(Stage D init + λ_traj=0.1) | **13.510** | — |
| L1(`..._L1_notraj`) | λ_traj=0(loss 關掉,init 保留) | 13.512 | +0.00 |
| L1b(`..._L1b_zeroinit`,新旗標 `--zero_traj_init`) | loss + init 全拔(Stage D 完全移除) | 13.528 | +0.02 |
| L2(`..._L2_p4`,新功能 `--n_parts 4`) | 多部位 Stage D:temporal-profile K-means → 4 條 per-part DLT 軌跡 + per-part λ_traj + per-cluster init | 13.570 | +0.06 |

**結論:整條階梯是平的(13.51–13.57)。Stage D / 軌跡 init / mask 粒度對 hellwarrior
的貢獻是零 —— I-1 的「單質心軌跡是主因」假說被推翻。** 瓶頸定位到 I-4 + handover §1c:
canonical pose(蹲)≠ SV4D t=0(站),超出 motion module 容量,photo/silh loss 自己就能
找到 Stage D 提供的那點資訊。

## 2. 對 lego 的反向 sanity(意外發現)

| 實驗 | lego_v2 PSNR vs clean |
|---|---:|
| control 重訓(`lego_v2_ctrl_rotfix`,Stage D 保留) | 20.603 |
| L1b(`lego_v2_L1b_zeroinit`,Stage D 全拔) | **20.790** |
| 舊 headline(leak-free+rotfix,供參) | 20.35 |

文件預測「lego 拿掉 Stage D 會退步」**也不成立** —— apples-to-apples(同 script 重訓)
下拔掉 Stage D 還 +0.19 dB。Stage E(photo + silhouette + ARAP)自己就能恢復動作。
(註:control 重訓 20.60 比舊 headline 20.35 高 0.25,應是訓練期 rotfix 的效果;
今後報數字建議都用本次重訓的 protocol。)
→ **方法敘事可以簡化:Stage D 從「必要組件」降級為「可選 init,對結果無影響」。**
(poster 的 5-stage pipeline 圖可保留,但 ablation 不用為 Stage D 辯護。)

## 3. L2 多部位 Stage D(已實作,泛用)

雖然對 hellwarrior 沒幫助,功能是好的、對未來多動件場景可用:
- `scripts/motion_parts_generic.py --n_parts P --out_suffix _xxx`:
  對 moving pixels 的 (T,)-dim temporal profile 做全域 K-means(timing 跨視角共享
  → 跨視角對應免費),per-part 加權質心 + per-part DLT,輸出
  `part_centroid_3d.npy (T,P+1,3)`、`gaussian_motion_part.npy (N,)`、per-view label maps。
- `scripts/train_partrigid_hier.py`:自動偵測多部位(centroid shape),per-cluster
  init 來自所屬 part 的軌跡(成員高斯多數決),λ_traj 改 per-part。
- 新旗標 `--zero_traj_init`(L1b 診斷用)。
- hellwarrior p4 sanity:4 條軌跡確實分開(位移 0.15–0.33、互距 0.14–0.28,conf~1.0)。

## 4. 3D-projection-prompted SAM-2(原型,可行 + 一張好圖)

`scripts/sam2_prompt_from_3d.py`(motionprior env):part 身分在 3D 上用 K-means 定一次
(左右肢在 3D 天然分開,結構上不可能 swap),投影到各視角(z-buffer 遮擋測試)當
SAM-2 的 point prompts;SAM 只修像素邊界、不決定身分。

發現:
1. **可行**:input view 左右手正確分開;側/背視角 part mask 解剖學上合理。
   跨視角身分一致性由「同一群 3D 高斯」保證,不依賴 SAM。
2. **第一版直接撞上 pose 不對齊**:背面視角 canonical(蹲)的腿投影落在背景上,
   SAM 把整片背景切出來。加「prompt 必須落在前景 alpha 內」的 guard 後修復。
   **`runs_aux/sam3dprompt_hellwarrior/overlay_v28_t00.png`(修復前版本)是
   pose-misalignment 最直觀的可視化證據** —— 比 13.5 dB 這個數字會說話,可考慮上 poster。
3. 剩餘縫隙(如預期):t>0 投影 prompt 跟著 canonical pose 走,不跟生成的動作走;
   要閉環需要用 Stage E 的粗動作 warp 投影(future work)。

輸出:`runs_aux/sam3dprompt_hellwarrior/overlay_v{00,14,28,42}_t{00,10}.png`
(左=raw 3D 投影,右=SAM refined)。

## 5. 對 framing 的含義

- **診斷主線不受影響**(probe 在 hellwarrior 本來就有效,ρ=0.87)。
- hellwarrior 重建差的根因敘事從「mask 太粗」改成「**canonical pose 不對齊,
  且我們證明了不是軌跡/mask 的問題(三個對照實驗,階梯全平)**」——這是更強的
  limitation 論述,poster 誠實邊界那格可以直接引用。
- 真要修 hellwarrior 重建,正路是:**給 canonical 一個 pose 對齊的來源**
  (站姿 D-NeRF 幀訓 canonical / 對 canonical 先做一次 global re-pose),
  或自產場景(jumpingjacks/standup 有乾淨 D-NeRF canonical)當第二個重建場景。

## 6.5 Mask-guidance 可視化套件(`scripts/viz_mask_guidance.py`,新)

釐清:**SAM-2 只用在 lego_v2(5-view)的 dataset alpha**(video predictor,
`sam2_mask_lego_v2.py`);**lego_v3/hellwarrior 完全沒用 SAM** —— alpha 是非白像素
門檻(`build_scene_dataset.py`,SV4D 背景本來就白)。兩者都餵進 `L_silh`,所以
guidance 檢驗方式相同。

每個 (view,t) 輸出 6 欄 panel:SV4D 監督幀 | gt alpha(silhouette 指導訊號)|
我們渲染的 alpha | disagreement(紅=leak 藍=miss,含 IoU)| Stage B motion mask |
變形後高斯按 part 上色投影。另輸出 IoU-vs-azimuth 曲線 + 表。

發現:
- **hellwarrior(L2 模型)**:input view IoU 0.83 → 偏軸 0.59–0.72,**又是 cone**
  (az=180 回升到 0.82,背面剪影對稱所致)。腿部 miss(藍)+邊緣 leak 環(紅)
  = pose 不對齊的形狀證據。Stage B motion mask 只抓到邊緣(低紋理深色軀幹的
  intensity variance 低)→ mask 確實粗,但 L1/L1b/L2 證明這不影響 PSNR。
- **lego_v2**:IoU 數字(均 0.41)被兩個已知 artefact 支配,不是 guidance 壞:
  (a) canonical 的底板高斯渲在 SAM digger-only mask 外(I-7,恆定 leak);
  (b) view 1 SV4D 自己只生成 bucket 特寫(D4 的 view-1 failure)。怪手本體
  silhouette 對得很好、arm 高斯正確聚在 bucket。
  → **silhouette IoU per view 本身又是一個 VGM 不一致 probe**(與 cone 同構)。

輸出:`runs_aux/mask_guidance_{hellwarrior_hellwarrior_cleancanon_L2_p4,lego_v2_lego_v2_ctrl_rotfix}/`
(panel_v*_t*.png、iou_vs_azimuth.png、iou_table.txt)。

## 5.5 後記(06-13 凌晨)— root cause 最終判定:是監督噪音,連 pose 都不是

第二輪判定實驗(同協議)推翻了 §5 的「canonical pose 不對齊」結論:

| 實驗 | 設定 | PSNR vs clean |
|---|---|---:|
| ctrl | SV4D 監督,binary gate | 13.51 |
| allmove | SV4D 監督,gate 全開(34579 顆全可動) | 13.38 |
| part-ARAP | SV4D 監督,跨 part ARAP ×0.1(`--arap_cross_part`) | 13.53 |
| **floor** | **乾淨 d-3dgs 監督,同 gate、同蹲姿 canonical** | **22.74** |

**floor = 22.74 證明:同樣的結構(binary gate + 蹲姿 canonical + 885K motion module)
在乾淨監督下 fit hellwarrior 毫無問題。** 所以 mask 粒度、軌跡、gate、joint 剛性、
canonical pose 全部排除 —— hellwarrior 的瓶頸是 **SV4D articulated 監督噪音本身**。

**Oracle gap = 第四個診斷儀器:**
- lego:oracle 20.96 − ours 20.35 = **0.6 dB**(rigid 內容,VGM 噪音傷害小)
- hellwarrior:oracle 22.74 − ours 13.51 = **9.2 dB**(articulated 內容,傷害大)
- 與診斷線三方互證:articulated temporal flicker 9.2×、FV4D 873 vs 471。
- → poster 敘事升級:「hellwarrior 重建差」不再是 limitation,而是
  **supervision-damage 量測的正面證據**(同一管線、同一 oracle 協議,
  per-scene 一個 scalar,GT 需求 = 只要乾淨參考訓 floor)。

MoE 化(routed rigid experts)與 fine-grained mask 的設計空間見
`docs/design/finegrained_mask_design.md` §1/§2/§6;依本判定,它們的價值在
故事乾淨度與診斷粒度,不在 hellwarrior 重建分數。

## 6. 本 session 產出的 artefacts

- 模型:`outputs/custom/partrigid_hellwarrior_cleancanon_{ctrl_rotfix,L1_notraj,L1b_zeroinit,L2_p4}`、
  `partrigid_lego_v2_{ctrl_rotfix,L1b_zeroinit}`
- 前處理:`runs_aux/part_assignment_hellwarrior_cleancanon_p4/`、`runs_aux/parts_motion_hellwarrior_cleancanon_p4/`
- 圖:`runs_aux/sam3dprompt_hellwarrior/overlay_*.png`
- 圖(06-13 補):`runs_aux/hellwarrior_gallery/` — 3-col gallery(SV4D|clean|ours,
  GIF×4 視角 + keyframe grid)+ **orbit_novel.gif**(novel-pose 軌道,
  floor 22.7 vs ours 13.5 並排 = oracle gap 的可視化,poster 候選)。
  `scripts/render_hellwarrior_gallery.py`
- log:`runs_aux/{train,eval}_hw_*.log`、`runs_aux/{train,eval}_lego_*.log`、`runs_aux/motion_parts_hw_p4.log`
- code:`motion_parts_generic.py`(--n_parts/--out_suffix)、`train_partrigid_hier.py`
  (--zero_traj_init/多部位)、`scripts/sam2_prompt_from_3d.py`(新)
