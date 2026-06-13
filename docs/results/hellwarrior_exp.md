# Hellwarrior 實驗總整理 — poster 素材頁

> 2026-06-13 整理。整合 06-12/13 root-cause 調查全部結果:假說階梯(7 組對照)、
> oracle-gap 判定、mask-guidance 可視化、3D-prompt SAM 原型、gallery/novel-view。
> 來源:`docs/planning/2026-06-12_hellwarrior_push.md`、`docs/design/finegrained_mask_design.md`。
> 圖檔:`runs_aux/hellwarrior_gallery/`、`runs_aux/mask_guidance_*/`、`runs_aux/sam3dprompt_hellwarrior/`。

---

## 1. 實驗設定

| 項目 | 內容 |
|---|---|
| 監督資料 | `hellwarrior`:SV4D 2.0 生成 57 視角(7 elev × 9 az)× 21 幀,576² |
| alpha/loss mask | **非白像素門檻**(`build_scene_dataset.py`;此場景沒用 SAM —— SV4D 背景本來就白) |
| 評估 GT | 獨立 Deformable-3DGS 訓在乾淨 D-NeRF hellwarrior(`hellwarrior_d3dgs_ref`) |
| Canonical | **乾淨** SC-GS static(`hellwarrior_scgs_default_node`,34,579 顆)——蹲姿(t=0 of D-NeRF),**與 SV4D t=0(站姿)不同 pose** |
| Ours | 凍結 canonical + K=100 SE(3)+LBS+xyz_res,leak-free motion-gated smart-photo |
| 協議 | 全部 rotfix(`--d_rot_zero` 訓練+評估)、8000 iters、`use_test_too` |
| 場景特性 | **articulated**(多肢同時動,全身蹲下)—— 與 lego(單臂 rigid)互補 |

跟 lego 的差異要點:57 視角(lego_v2 是 5)、無 SAM mask、canonical pose 不對齊、
全身動(Stage C 投票只標 35.9% 高斯為 moving)。

---

## 2. 假說階梯:重建為什麼卡在 13.5 dB(root-cause 排除實驗)

七組對照、同協議,系統性排除所有「模型/mask」假說:

| # | 假說 | 實驗(label) | PSNR vs clean | 判定 |
|---|---|---|---:|---|
| 1 | (基準) | ctrl(`..._ctrl_rotfix`) | 13.51 | — |
| 2 | λ_traj loss 拉壞 | L1(`..._L1_notraj`) | 13.51 | ❌ 排除 |
| 3 | Stage D 軌跡 init 錯(open_issues I-1) | L1b(`--zero_traj_init`) | 13.53 | ❌ 排除 |
| 4 | mask 太粗/單質心軌跡 | L2(`--n_parts 4`,temporal-profile 多部位 Stage D) | 13.57 | ❌ 排除 |
| 5 | binary gate 鎖死全身動作 | allmove(gate 全開,34,579 顆全可動) | 13.38 | ❌ 排除 |
| 6 | ARAP 把關節焊死 | part-ARAP(`--arap_cross_part 0.1`) | 13.53 | ❌ 排除 |
| 7 | canonical pose 不對齊(I-4) | **floor:同蹲姿 canonical、同 gate,換乾淨 d-3dgs 監督** | **22.74** | ❌ **連這個都排除** |

**結論:#1–#6 整條階梯是平的(13.4–13.6);#7 證明同樣的結構在乾淨監督下
fit hellwarrior 毫無問題。瓶頸 = SV4D 在 articulated 內容上的監督噪音本身。**

lego 反向 sanity(同協議重訓):Stage D 全拔 20.79 ≥ 保留 20.60 ——
**Stage D 在兩個場景都是可選的**,方法敘事可瘦身(Stage E 光靠
photo+silh+ARAP 就能找到動作)。

附:binary gate 其實是「軟」的 —— static 側高斯 median arm_weight 0.263
(可跟 26% 的 cluster 動作 + 受正則化的 xyz residual),但 allmove 證明
全開也沒差,gate 軟硬不是變因。

---

## 2.7 vs vanilla SC-GS(joint)— 與 lego 對稱的 baseline,但結論相反且重要

> 補上 lego 有、hellwarrior 缺的 vanilla 對照。同 SC-GS 配方(joint、512 nodes、
> 16M deform-MLP、20k iters)訓在 hellwarrior 的 57 視角 SV4D 監督。
> 腳本:`scripts/render_hw_vanilla_compare.py`。

> 全部數字用**同一條渲染管線**(`eval_region_psnr` renderer + 前景合成 PSNR,
> 同樣用於 ours 與 lego vanilla),確保可比。

| 場景 / baseline | vanilla vs clean | ours vs clean | vanilla vs SV4D | ours vs SV4D | Δ vs clean |
|---|---:|---:|---:|---:|---:|
| **lego_v2(5 視角,rigid)** | 11.43(幾何爆炸) | 20.35 | — | — | **+8.9** |
| **hellwarrior(57 視角,articulated)** | **12.68** | 13.51 | 13.68 | 15.31 | **+0.83** |

**關鍵:57 視角的 vanilla 不像 5 視角那樣「完全尖刺崩塌」,但也遠非乾淨**——
身體輪廓還在(多視角約束多),但整體是模糊煙霧,vs-clean 只有 12.68。ours 因
凍結結構更貼身體、且 vs-SV4D(15.31)> vanilla(13.68)= **去噪而非過擬合**,
但領先只有 +0.83 vs clean,**兩者都遠離 clean GT(floor 22.7)**。

> ⚠️ **更正(2026-06-13)**:SC-GS 訓練 log 內部報的 "Best PSNR 25.46"
> **不可採信為對照數字** —— 那是 SC-GS 自己的 eval 路徑(疑似含白背景灌水/
> 不同 test 定義),與我們前景合成的一致管線差 ~12 dB。一律以上表的一致管線為準。

vanilla vs ours(每列:SV4D | clean GT | vanilla | ours;標 vs-clean PSNR):

![](../../runs_aux/hellwarrior_vanilla_compare/vanilla_vs_ours_keyframes.png)

vanilla vs 自己的 SV4D 監督(held-out 時刻;證明身體輪廓在、非全崩,但仍模糊 ~13.5 dB):

![](../../runs_aux/hellwarrior_vanilla_compare/vanilla_fits_sv4d_heldout.png)

動態 ours vs vanilla(SV4D | vanilla | ours)+ novel view(相機在 grid 方位之間插值):

![](../../runs_aux/scene_videos/hellwarrior_compare_v0.gif)

![](../../runs_aux/scene_videos/hellwarrior_novel.gif)

訓練曲線(8000 iters;loss 抖動因每步隨機抽 view/t):

![](../../runs_aux/train_curve_hellwarrior.png)

**讀法(誠實,且強化中心論點):**
- **方法優勢隨監督噪音增大而縮小**:rigid+欠約束(lego_v2)method 差 +8.9 dB;
  articulated+多視角(hellwarrior)只差 +0.83 dB。
- **原因正是 oracle gap**:hellwarrior 的瓶頸是監督噪音(§3 floor 22.7),
  vanilla 與 ours **撞同一道數據天花板**(12.7 / 13.5,都遠低於 22.7)。
  ours 的 vs-SV4D > vanilla = 凍結結構在去噪;但噪音上限壓過一切。
- **這不是 hellwarrior 重建方法失敗,而是 pivot 的正當性證據**:當監督夠髒,
  「換更好的 reconstructor」收益趨近於零 → 該做的是**診斷 generator**。
  method-advantage 隨噪音衰減本身 = 又一個 supervision-damage 訊號。

**Caveat**:lego 的 11.43 來自 5 視角 lego_v2,與 hellwarrior 57 視角不同視角數。
嚴格 apples-to-apples 需在 lego_v3(57 視角)也跑 vanilla(預期不全崩、gap 縮小)。

---

## 3. Oracle gap:重建失敗變成第四個診斷儀器(poster 主打)

同一管線、同一 oracle 協議(ours 訓在乾淨參考 = floor/ceiling),per-scene 一個 scalar:

| 場景 | oracle(乾淨監督) | ours(SV4D 監督) | **oracle gap** | overfit gap(vs-SV4D − vs-clean) |
|---|---:|---:|---:|---:|
| lego(rigid) | 20.96 | 20.35 | **0.6 dB** | **+6.0**(在去噪) |
| hellwarrior(articulated) | 22.74 | 13.51 | **9.2 dB** | **−1.8**(在 fit 噪音) |

兩個讀法,都是診斷線的正面證據:
- **oracle gap = supervision damage**:同一套方法,rigid 內容只損失 0.6 dB,
  articulated 內容損失 9.2 dB。與診斷線三方互證:articulated temporal flicker
  9.2×(D3 generality)、FV4D 873 vs 471(遮底板)。
- **overfit gap 變號**:lego 上我們的輸出比監督更接近乾淨 GT(+6.0,管線在
  去噪);hellwarrior 上變號(−1.8,噪音強到只能 fit 噪音)—— 「smart-photo
  能濾掉的不一致」有一個量級上限,articulated flicker 超過了它。

**敘事升級:hellwarrior 重建差不再是 limitation,而是量測結果** ——
「凍結式管線把 VGM 的 articulated 失敗模式轉成 9.2 dB 的可量化傷害」。

### Perceptual metric 全表(PSNR/SSIM/LPIPS,雙 GT)— 2026-06-13 補

> 1197 frames(全 57 視角)、LPIPS=AlexNet(低=好)。
> 腳本:`scripts/eval_metrics_table.py` → `runs_aux/metrics_table_hellwarrior/`。

| model | PSNR vs clean | SSIM vs clean | LPIPS vs clean | PSNR vs SV4D | LPIPS vs SV4D | overfit gap (dB) |
|---|---:|---:|---:|---:|---:|---:|
| ours(SV4D 監督) | 13.51 | 0.824 | 0.240 | 15.31 | 0.217 | −1.80 |
| oracle(乾淨監督,同結構) | 22.75 | 0.938 | **0.091** | 12.95 | 0.230 | +9.79 |

**讀法 —— oracle gap 不依賴 PSNR,三個 metric 同向:**

| oracle gap 的版本 | lego(rigid) | hellwarrior(articulated) |
|---|---:|---:|
| PSNR(oracle − ours) | 0.4–0.6 dB | **9.2 dB** |
| LPIPS(ours ÷ oracle) | 0.92×(無損傷,ours 略佳) | **2.6×** |
| SSIM(oracle − ours) | ~0.00 | **0.114** |

- LPIPS 0.091 同時證明 oracle 的 22.7 dB 不是模糊高分 —— 乾淨監督下知覺品質
  真的好(對照 lego ours 0.146)。
- overfit gap:lego ours +5.95(去噪)vs hellwarrior ours −1.80(fit 噪音)——
  變號結論在 perceptual 協議下成立。

## 3.5 嘗試 refine 重建:reliability-weighted supervision(2026-06-13)

> 目標:讓 hellwarrior 重建追平 lego。**唯一對症的方法槓桿** —— 既然瓶頸是
> SV4D 偏軸噪音(§2),就用診斷結果反過來加權監督信任(poster takeaway #1
> 的直接實作):per-view photometric 信任 = exp(−β·(az_dist/180 + 0.5·elev/30)),
> 壓低偏軸噪聲視角,讓 part-rigid 剛性結構從可靠視角外推。旗標
> `--view_reliability_beta`(+`--view_reliability_on_silh`)。

| 變體 | mean PSNR vs clean | Δ vs control | 備註 |
|---|---:|---:|---|
| control | 13.51 | — | |
| β=1.5(photo) | 13.82 | +0.31 | |
| β=3(photo) | 13.92 | +0.41 | |
| **β=5(photo)** | **13.97** | **+0.46** | 最佳,已飽和 |
| β=3 + silhouette 也加權 | 13.62 | +0.11 | silhouette 是乾淨幾何錨,壓它反而傷 |
| β=3 + 強時間平滑 | 13.91 | +0.40 | flicker 在監督裡不在模型抖動,平滑無增益 |

**Per-azimuth 分解**(t 採樣 0/5/10/15/20,`compare_hellwarrior_variants.py`):

| az_dist | control | β=3 | β=5 |
|---:|---:|---:|---:|
| 0(input) | 15.25 | 15.83 | 15.91 |
| 60 | 14.31 | 14.75 | 14.82 |
| 90 | 13.14 | 13.53 | 13.56 |
| 150 | 13.18 | 13.58 | 13.63 |
| 180 | 14.39 | 14.74 | 14.81 |
| **可靠錐(≤60)** | 14.11 | 14.59 | **14.68** |
| **遠端(≥150)** | 13.59 | 13.96 | **14.03** |

**判定(誠實):**
- **方法原理被驗證**:壓低偏軸監督,平均反而**升** → 偏軸 SV4D 監督是**淨有害**
  (模型在 fit 噪音);trust-by-view 是對的。這是 poster takeaway #1 第一次有
  *重建端* 的實驗支撐(之前只是 loss 設計)。
- **但增益飽和在 +0.5 dB,且幾乎均勻**(可靠錐 +0.57、遠端 +0.44)——
  **不是「可靠錐追平 lego」的戲劇效果**。
- **數據天花板才是 binding constraint**:floor 22.7(乾淨監督)vs 任何監督加權
  能達到的 ~14.0 —— 缺的 8.7 dB 是 SV4D 影片裡**根本不存在的資訊**,任何
  *方法*(mask/gate/MoE/ARAP/reliability/temporal)都救不回來。已窮舉測試。

**結論:純方法無法讓 hellwarrior 看起來像 lego。** 視覺對比(下圖):β=5 比
control 略乾淨(fuzz 稍收),但仍是模糊團,離 lego/clean GT 很遠。

![](../../runs_aux/hellwarrior_variant_compare/variant_keyframes.png)

![](../../runs_aux/hellwarrior_variant_compare/per_azimuth_psnr.png)

**要真正得到 lego 級 articulated 重建,只有兩條路**(都不是改 method):
1. **更乾淨的 generator**(SV4D 的 articulated flicker 是根因 → future work)。
2. **自產場景**(jumpingjacks/standup 有乾淨 D-NeRF canonical,且我們控制生成)——
   articulated 但監督可能較乾淨,是「articulated 也能重建」的最快候選。

→ 對 poster:reliability-weighting 收進方法線當「診斷→重建閉環」的 +0.5 dB
小證據(takeaway #1 落地),hellwarrior 的角色仍是 §3 的 supervision-damage 量測。

---

### Oracle gap 的可視化(novel pose 軌道,poster 候選圖)

相機沿 elev-0 ring 在 grid 之外的方位角 slerp(novel pose)、動作同步播放;
左 = floor(乾淨監督)、右 = ours(SV4D 監督)。**同 canonical、同動作模組、
同 pose,唯一差別是監督乾淨度:**

![](../../runs_aux/hellwarrior_gallery/orbit_frame05.png)

動畫:`runs_aux/hellwarrior_gallery/orbit_novel.gif`(21 幀)。
腳本:`scripts/render_hellwarrior_gallery.py`。

### 3-col gallery(訓練視角,SV4D | clean GT | ours)

![](../../runs_aux/hellwarrior_gallery/gallery_keyframes.png)

- 第 1 欄 vs 第 2 欄本身就展示 SV4D 的 pose 偏移(同 (view,t),姿態不同)。
- 第 3 欄的針狀高斯霧 = 噪音「顯形」(凍結結構,噪音無處可藏 —— 故事第 3 幕
  在 articulated 場景的極端版)。
- 逐視角 GIF:`runs_aux/hellwarrior_gallery/gallery_v{00,07,14,28}.gif`。

---

## 4. Mask / guidance 可視化(回答「高斯有被好好 guide 嗎」)

`scripts/viz_mask_guidance.py`:每 (view,t) 一張 6 欄 panel
(監督幀 | gt alpha | 渲染 alpha | disagreement | Stage B motion mask | 高斯 part 投影)
+ 全 57 視角 silhouette-IoU 掃描。

![](../../runs_aux/mask_guidance_hellwarrior_hellwarrior_cleancanon_L2_p4/panel_v00_t10.png)

- input view IoU 0.83,本體跟得住;腿部藍色 miss + 邊緣紅色 leak 環 =
  pose 偏移的形狀證據。
- Stage B motion mask 只抓到邊緣(深色低紋理軀幹 intensity variance 低)——
  mask 確實粗,但 §2 階梯證明這不影響 PSNR。

**Silhouette IoU vs azimuth 又是一個 cone**(az=0 最高 0.83 → 偏軸 0.59;
az=180 因背面剪影對稱回升):

![](../../runs_aux/mask_guidance_hellwarrior_hellwarrior_cleancanon_L2_p4/iou_vs_azimuth.png)

→ 純形狀、GT-free、與外觀無關的第五個 cone 重現(若三儀器表想加一行,免費)。

### 3D-projection-prompted SAM-2(原型,`scripts/sam2_prompt_from_3d.py`)

part 身分在 3D 上定一次(canonical K-means,左右肢天然分開、結構上不可能
swap)→ 投影(z-buffer 遮擋)當 SAM-2 point prompts,SAM 只修邊界:

![](../../runs_aux/sam3dprompt_hellwarrior/overlay_v00_t00.png)

- input view 左右手正確分開(綠/黃),跨視角身分由同一群 3D 高斯保證。
- 第一版在背面視角撞上 pose 不對齊(蹲姿的腿投影落到背景,SAM 切了整片
  背景)—— 加前景 guard 後修復;失敗版本本身是 pose 偏移最直觀的可視化。
- 尚未進訓練 loop;閉環需要把投影跟著 Stage E 粗動作 warp(future work)。

---

## 5. 與診斷主線的關係

重建卡 13.5 **不影響**診斷線,反而強化:
- fit-residual probe 在 hellwarrior 照樣重現 cone(ρ=0.87,GT-free)——
  fitting 吸收 registration,probe 對重建品質不敏感(這正是賣點)。
- generality:rigid → 空間崩、articulated → 時間 flicker 主導(9.2×)。
- 本頁 §3 的 oracle gap + overfit-gap 變號是診斷的第四、第五個獨立訊號。

GAP-2(R_clean 平坦 → cone 屬於 SV4D 不是參考,hellwarrior 在右):

![](../../runs_aux/gap2_reference_floor.png)

獨立儀器(SED 幾何 + FVD/FV4D 分佈,hellwarrior 全項比 lego 差):

![](../../runs_aux/instruments_sed_fv4d.png)

---

## 6. Poster 怎麼引用(建議措辭)

**Claim:**「同一套凍結式管線,rigid 內容只被 VGM 噪音傷 0.6 dB,articulated
內容被傷 9.2 dB —— 與我們量到的 9.2× articulated 時間 flicker 互相印證;
重建端與診斷端給出同一張失敗地圖。」

- 主數字:**oracle gap 0.6 vs 9.2 dB**(配 §3 orbit 對比圖或 GIF)。
- 輔助:overfit gap 變號(+6.0 → −1.8:去噪 → fit 噪音);probe ρ=0.87 不受影響。
- 防禦(被問「是不是你們模型爛」):§2 七組對照階梯 —— 軌跡/mask/gate/
  關節/canonical pose 全部排除,floor 22.74 證明容量足夠。
- **不 claim** hellwarrior 重建品質;它的角色是 articulated 失敗模式的量測證據。

### 數字一致性備註
- 本頁全部用 rotfix-retrain 協議(ctrl 13.51);舊文件的 13.53(pre-rotfix
  cleancanon)、15.48(SV4D-trained canonical,full-57)是不同協議/不同
  canonical 的歷史數字,不要混表。
- lego oracle gap 用 20.96−20.35(同為舊協議對);hellwarrior 用
  22.74−13.51(同為 rotfix 協議對)—— 各自 internally consistent。
- §3 gallery 標的 per-view PSNR(如 v0 15.0)是單視角均值,天然高於全 57 視角
  均值 13.51,不要跨表比大小。
