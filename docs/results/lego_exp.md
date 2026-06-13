# Lego 重建實驗總整理 — poster 素材頁

> 2026-06-12 整理。整合三個分析:region-decomposed PSNR(「贏在底板?」)、
> novel-view 幾何對比、SC-GS 實作 sanity check,加上資料格式說明。
> 來源報告:`docs/reports/2026-06-12_region_psnr.md`、`2026-06-12_novel_view_check.md`。
> 圖檔:`runs_aux/novel_view_compare/`、`meetTW_checkpoint_0601/figs/`。

---

## 1. 實驗設定

| 項目 | 內容 |
|---|---|
| 監督資料 | `lego_v2`:SV4D 2.0 生成的 5 視角(elev0,az 0/60/120/180/240)× 21 幀,576² |
| alpha/loss mask | SAM-2 video-predictor digger-only mask(**底板不在監督裡**) |
| 評估 GT | 獨立 Deformable-3DGS 訓在乾淨 D-NeRF lego,渲在相同相機(`lego_v2_d3dgs_ref`) |
| Vanilla baseline | SC-GS joint(random init + 16M deform-MLP),20k iters |
| F1 / F2 | clean canonical 給 SC-GS:F1 = warm-start(不凍結)、F2 = 凍結 + deform-MLP |
| Ours | 凍結 canonical + cluster SE(3)+LBS+xyz_res(885K 參數),leak-free smart-photo |
| Novel view | lego_v3 57-view grid 中不在訓練 5 視角的 pose(world frame 已驗證完全一致) |

### 資料格式:跟 D-NeRF 一樣嗎?

格式相容(SC-GS 用 `--is_blender` 原樣讀),語意不同:

| | D-NeRF lego | 我們(lego_v2) |
|---|---|---|
| JSON | blender 格式 | **同格式** + 額外 key(view_idx 等,SC-GS 忽略) |
| 相機協議 | 單目 teleporting:50 幀,每幀獨特 (pose, time) | **多視角 grid**:5 固定 view × 21 共享 timestep |
| 相機來源 | Blender GT pose | SV4D 可控軌道相機,數學推導(非 SfM) |
| 內參 | camera_angle_x 0.6911 | **相同 0.6911**(刻意對齊 canonical 來源) |
| 解析度 / alpha | 800²,真 alpha(含底板) | 576²,**SAM-2 binary digger mask** |
| split | 不同 (pose,time) | **temporal**(t%4==0 → test;視角兩 split 共用) |

處理鏈:SV4D mp4(每視角)→ 解碼 21 幀 → 軌道相機數學轉 c2w → SAM-2 propagate 出 digger mask 當 alpha(`scripts/sam2_mask_lego_v2.py`)。新場景一鍵建:`scripts/build_scene_dataset.py`(alpha 用非白 threshold)。

注意:官方 test split 是 **held-out time 不是 held-out view** → 下面 §3 的 novel view 才是真正的 unseen-view 檢驗。

---

## 1.5 方法 component ablation(我們特別設計的部件,各貢獻多少)

我們的方法 = **凍結乾淨 canonical + part-rigid 動作模組**。逐個 component(全在 lego_v2 上做):

| # | Component(設計部件) | 做什麼 / 為什麼 | 貢獻 |
|---|---|---|---|
| A | **凍結乾淨 canonical**(Stage A) | 結構 `requires_grad=False`,噪音無法吸進幾何 → 解耦 | **最大單一貢獻**(vanilla 11.43 → 15.91 靜態,+4.5) |
| B | **Motion mask + part 投票**(Stage B/C) | 時間變異 + Otsu + 多視角投票 → arm/body 二分,只讓動的部位動 | 限制動作 DOF(防止 body 吸噪) |
| C | **K=100 cluster SE(3) + LBS**(Stage E 核心) | arm 高斯 K-means 100 群,每群 per-time 剛體 SE(3);LBS top-6 軟混合 | K sweep 甜蜜點(K=10→100 +0.69;K=200/300 反而過擬合噪音) |
| D | **Smart-photo loss**(VGM 信心加權) | `w=exp(−α·\|SV4D−canon\|)`,不一致像素權重→0 濾掉 hallucination | **+1.42**(最大 loss 機制) |
| E | **Per-time per-cluster scale** | 每群每時刻 3D scale,修 canonical 形狀掃動時的 streaking | +0.22 |
| F | **Per-Gaussian XYZ residual** | cluster SE(3) 是剛體;residual 給連續非剛性微修(重正則防過擬合) | **+0.44**(motion module 內最大單機制) |
| G | Stage D 軌跡 init(DLT) | 質心三角化當 SE(3) 平移 init | 可選(hellwarrior/lego 拔掉無損,見各 exp) |
| — | ~~Per-Gaussian rot residual~~ | 試過每高斯旋轉殘差 | **+0.04 → 拒絕**(誠實負結果) |

### Component B 長怎樣(Stage B motion mask + Stage C part 投票)

**怎麼算**:Stage B = 每視角逐像素**時間標準差**(across 21 幀)+ Otsu 門檻(前景內)
+ 形態學清理 → 「在動」的像素。Stage C = 把 canonical 高斯投影到各視角,用 motion mask
**多視角投票** → 每顆高斯 arm / body。**純啟發式、零學習、無光流。**

lego(單一動件,乾淨):紅 = motion mask / arm 高斯,集中在鏟斗臂;車身履帶底板 = body:

![](../../runs_aux/stageBC_lego_v2.png)

jumpingjacks(全身動,粗糙):紅區大且溢出輪廓 —— mask 明顯粗:

![](../../runs_aux/stageBC_jumpingjacks.png)

**這真的有用嗎?(誠實)** mask 很粗,但**精度不重要** —— hellwarrior 的 allmove 實驗
(gate 全開、等於不要 mask)= 13.38 vs 用 mask 13.51,幾乎一樣。mask 的**真正功能**
不是精確分割,而是當 **smart-photo 的開關**:smart-photo 拿靜態 canonical 當參考,
若對動的像素也濾波,動區永遠跟靜態 canonical 不一致 → 權重→0 → **動作學不到**;
mask 只需粗略二元「這塊動不動」來決定動區關掉濾波。→ **B 是讓 D(smart-photo)能用的
輔助件,不是 +dB 主力**(主力是 A 凍結 canonical + D smart-photo)。

累積階梯(左)+ 各機制單獨貢獻(右,含被拒絕的 rot-residual):

![](../../runs_aux/component_ablation.png)

視覺進展(SV4D | clean GT | K=1 單剛體 | K=100+LBS+smart | +xyz residual 全配;
手臂從僵硬剛體 → 多群關節 → 連續微修,愈來愈貼 GT):

![](../../runs_aux/component_ablation_visual.png)

**讀法:**
- **凍結 canonical 是骨幹**(+4.5,最大跳),其餘是動作模組的精修。
- **贏在 inductive bias 不是 DOF**:同 canonical 換 16M deform-MLP(F2)= 11.89,
  我們 885K = 20.03 → 結構化 SE(3)+LBS 才是關鍵(見 §2 canonical 2×2 排除實驗)。
- **誠實負結果**:rot-residual(+0.04)、更鬆的 xyz_res 正則(−0.19)都試過並拒絕。

---

## 2. 訓練視角:region-decomposed PSNR(「贏在底板?」的定量回答)

全幀 PSNR 有區域效應:83% 像素是背景、底板只有我們有(從凍結 canonical 繼承,監督裡沒有)。同一組 mask 套所有模型,只算 mask 內像素:

| 模型 | full | **digger**(7.6% px) | baseplate(9.5%) | background(82.9%) |
|---|---:|---:|---:|---:|
| Vanilla SC-GS | 11.43 | 10.66 | 5.27 | 13.64 |
| F1 warm-start | 11.55 | 10.87 | 5.42 | 13.70 |
| F2 frozen + deform-MLP | 11.89 | **3.59** | 4.81 | **62.91** |
| **Ours** | **20.64** | **14.90** | **14.70** | **24.80** |

(腳本 `scripts/eval_region_psnr.py`;baseline 全幀數字與歷史報告精準重現。)

**讀法:**
- 質疑部分成立:全幀 +8.6/9.2 dB 確實被底板+背景放大;**apples-to-apples 的 digger 區我們贏 +4.2 dB**。
- **F2 的「+0.46 贏 vanilla」是假象**:它物體區 3.59 dB(deform-MLP diverge,物體渲不出來),全幀分數全靠白背景(62.9 dB)。→ fairness 論證更強:同一顆凍結 canonical,deform-MLP 物體直接崩,我們 14.9 dB。
- 底板 14.7 dB 來自「body part 不動」的歸納偏置 —— 是 claim 本身,但要跟 digger 數字分開報。

### Canonical 給法 2×2 排除實驗(「freeze 造成矛盾訊號?」)

質疑:把 clean canonical 塞給 SC-GS 並 freeze,會不會反而給 deform-MLP 矛盾訊號才破圖?
排除法:不給(vanilla)/ 給不凍(F1)/ 給且凍(F2)三格全跑過 ——
**爆炸在「不給」和「給不凍」就存在**;freeze 只是把失敗型態從爆炸換成 diverge。破圖源頭是 noisy 監督,不是 canonical 的給法。

每列 = clean GT | vanilla | F1(不凍)| F2(凍+MLP)| ours;上列訓練視角 az120、下列 novel view az270(t=10):

![](../../runs_aux/novel_view_compare/canonical_ablation_4model.png)

- vanilla 與 F1 幾乎同樣的尖刺爆炸(F1 的 warm-start 在 1–2k iter 被 joint training 洗掉)。
- F2 **全視角渲染近乎空白**(它的 11.9 dB 全靠白背景)。
- 同一顆凍結 canonical,唯一差別是動作模組 → ours 幾何正確。**Freeze 本身無罪,缺的是動作的結構性約束。**

#### F2「消失」的機制解剖

物體 bbox 僅 ~1.5 單位寬,但 F2 的 deform-MLP 在**所有時刻**(含 t=0)輸出
中位數 13.6–14.1 單位的位移 → 114k 顆高斯幾乎整體被搬出相機視錐:

![](../../runs_aux/novel_view_compare/f2_gaussian_drift.png)

(藍 = canonical;紅 = F2 變形後,±3 單位內只剩碎片。
多視角空白渲染證據:`runs_aux/novel_view_compare/f2_views_grid.png`。)

**把逃逸的高斯找回來**(用 look-at 追蹤相機對準逃逸雲):

全範圍散點 —— 不是炸開,是**整體同向漂移**(方向一致性 |mean unit|=0.999,
方向 ≈ (0,−0.36,−0.93) 往下偏後;centroid 在 (−0.4,−5.1,−13.1),
沿逃逸方向拉長成 ~12 單位鏈狀):

![](../../runs_aux/novel_view_compare/f2_escape_fullextent.png)

追蹤相機實拍 —— 挖土機的黃色車體/棕色履帶還認得出來,被抹成一串巨大模糊泡泡。
屬性凍結所以顏色逃不掉,**叛逃的只有位移場**:

![](../../runs_aux/novel_view_compare/f2_chase_cam.png)

為什麼:(1) 高斯一旦離開所有視錐,photometric/silhouette 梯度歸零 ——
**逃逸是吸收態**;(2) 方向一致 = 這是「全域平移」這個 ARAP 零成本模式
(ARAP 只罰相對形變)被矛盾監督的發散梯度推著走;(3) 訓練 loss
0.5→4.0 發散與此一致。我們的 SE(3)+LBS 為何免疫:per-cluster 6 DOF +
軌跡 anchor(質心被 DLT 三角化目標拉住),「整團搬走」的方向被 anchor
直接罰掉。

**Caveat(誠實邊界):** F2 是 diverged run(調 lr/node init 或許可救);
結論不依賴 F2 —— F1/vanilla 用 SC-GS 上游預設 joint 配方照樣爆炸。
F2 的價值是展示「凍結 + 無結構動作」的失敗型態。

訓練視角質性對比(SV4D | clean GT | 方法;v0 t=10):

| Vanilla(爆炸) | Ours(糊但完整) |
|---|---|
| ![](../../meetTW_checkpoint_0601/figs/vanilla_v0_t10.png) | ![](../../meetTW_checkpoint_0601/figs/ours_v0_t10.png) |

動態 ours vs vanilla(SV4D | vanilla 黑尖刺爆炸 | ours;21 幀)+ novel view orbit:

![](../../runs_aux/scene_videos/lego_v2_compare_v0.gif)

![](../../runs_aux/scene_videos/lego_v2_novel.gif)

訓練曲線(8000 iters;loss 抖動因每步隨機抽 view/t):

![](../../runs_aux/train_curve_lego.png)

---

## 2.5 Perceptual metric 全表(PSNR/SSIM/LPIPS,雙 GT)— 2026-06-13 補

> 補「PSNR 獎勵模糊」的漏洞。全幀、105 frames、同協議;LPIPS=AlexNet(低=好)。
> **overfit gap = vs-clean − vs-SV4D**(正 = 輸出比自己的監督更接近乾淨 GT = 在去噪)。
> 腳本:`scripts/eval_metrics_table.py` → `runs_aux/metrics_table_lego_v2/`。

| model | PSNR vs clean | SSIM vs clean | LPIPS vs clean | PSNR vs SV4D | LPIPS vs SV4D | overfit gap (dB) |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 11.43 | 0.776 | 0.289 | 12.84 | 0.190 | −1.41 |
| F1 warm-start | 11.55 | 0.771 | 0.297 | 12.84 | 0.202 | −1.29 |
| F2 frozen+MLP | 11.89 | 0.839 | 0.295 | 15.50 | 0.177 | −3.61 |
| **ours** | **20.60** | **0.855** | **0.146** | 14.65 | 0.247 | **+5.95** |
| ours(無 Stage D,L1b) | 20.79 | 0.860 | **0.125** | 14.72 | 0.228 | +6.07 |
| oracle(訓乾淨 GT) | 20.96 | 0.856 | 0.158 | 14.46 | 0.254 | +6.50 |

**讀法:**
- **LPIPS 背書「+8.6 dB 不是模糊灌的」**:ours 0.146 vs vanilla 0.289(知覺品質 2×);SSIM 同向(0.855 vs 0.776)。質疑正式關閉。
- **overfit gap 一欄講完去噪故事**:三個 baseline 全是負值(輸出更像噪音監督),ours +5.95(輸出比自己的監督更接近乾淨 GT)。
- **無 Stage D 版 LPIPS 0.125 全場最佳、連 oracle(0.158)都贏** —— Stage D 拿掉後知覺品質最好(oracle 是 pre-rotfix checkpoint,跨協議比較留意)。
- F2 的 SSIM 0.839 與 vs-SV4D 15.50 偏高仍是白背景灌水(§2 region 分析:物體區 3.59 dB),不要單看。

---

## 3. Novel view:幾何正確性對比(poster 主打)

兩模型都只看過 5 視角;渲在 6 個 unseen pose,GT = 乾淨 d-3dgs:

| novel view | vanilla | ours | Δ |
|---|---:|---:|---:|
| elev0 az150 | 11.55 | 19.46 | +7.9 |
| elev0 az210 | 11.70 | 21.29 | +9.6 |
| elev0 az270 | 10.74 | 19.02 | +8.3 |
| elev0 az330 | 11.10 | 18.57 | +7.5 |
| elev20 az180 | 11.03 | 20.30 | +9.3 |
| elev30 az120 | 11.17 | 19.82 | +8.7 |
| **mean** | **11.22** | **19.74** | **+8.5** |

每列 = clean GT | vanilla SC-GS | ours(t=10):

![](../../runs_aux/novel_view_compare/novel_keyframe_grid.png)

逐視角 21 幀動畫:`runs_aux/novel_view_compare/novel_elev_*_az_*.gif`(6 支,poster QR/digital 版可用)。

### 四模型版(含 F1/F2,canonical 排除實驗的 novel view 延伸)

| novel view | vanilla | F1 不凍 | F2 凍+MLP | ours |
|---|---:|---:|---:|---:|
| elev0 az150 | 11.55 | 12.09 | 11.74 | **19.46** |
| elev0 az210 | 11.70 | 11.93 | 11.83 | **21.29** |
| elev0 az270 | 10.74 | 10.98 | 10.39 | **19.02** |
| elev0 az330 | 11.10 | 11.38 | 10.66 | **18.57** |
| elev20 az180 | 11.03 | 11.35 | 10.58 | **20.30** |
| elev30 az120 | 11.17 | 11.02 | 11.05 | **19.82** |
| **mean** | 11.21 | 11.46 | 11.04 | **19.74** |

每列 = clean GT | vanilla | F1 | F2 | ours(t=10):

![](../../runs_aux/novel_view_compare/novel4_keyframe_grid.png)

逐視角動畫:`runs_aux/novel_view_compare/novel4_*.gif`(6 支)。

- **三個 baseline 在 novel view 全部 ~11 dB**,差異在誤差棒內 —— canonical 怎麼給都救不了 deform-MLP。
- F1(不凍)跟 vanilla 同樣的爆炸碎片 → 排除「freeze 造成破圖」。
- **F2(凍)在所有視角(含訓練視角)渲染近乎空白** —— 位移場把高斯整體搬出視錐(機制解剖見 §2);凍結屬性擋不住 16M 自由度的位移場。
- ours 全視角幾何正確 → novel view 是分離「動作模組約束」貢獻最乾淨的軸。

**讀法:**
- vanilla 在每個 novel view 都是黑色爆炸碎片(az210/270 幾乎只剩殘片)—— 不是品質差,是**幾何不存在**。
- ours 所有視角位置/姿態/底板正確,只輸銳利度。
- **view-uniform**:我們 novel(19.7)≈ 訓練視角(20.6);vanilla 訓練視角(11.4)≈ novel(11.2)—— 它連訓練視角都是過擬合碎片,沒有 3D 幾何可言。

---

## 4. Sanity check:SC-GS 實作沒被我們弄壞

同一份 SC-GS 程式 + 同樣旗標,訓在**乾淨 D-NeRF lego**(20k iters):

- held-out test:**PSNR 25.18 / SSIM 0.950 / LPIPS 0.053** —— 正常 SC-GS 水準。
- 肉眼:幾何完整、零爆炸(5 個 test view × 不同時刻,GT | render):

![](../../runs_aux/novel_view_compare/dnerf_lego_sanity_grid.png)

模型:`outputs/custom/dnerf_lego_vanilla_sanity_node/`。

**歸因鏈(回答「baseline 是不是被你們弄壞」):**

| 檢查 | 結果 |
|---|---|
| SC-GS × 乾淨 D-NeRF lego(1 view/time) | 25.2 dB,正常 → 實作 OK |
| SC-GS × SV4D lego_v2(5 views/time,noisy) | 11.4 dB,爆炸 → 崩壞來自 noisy 監督 |
| Ours × 同樣 SV4D 監督,novel view | 19.7 dB,幾何正確 → **+8.5 dB 是方法差距** |

附:D-NeRF 每 time 只有 1 視角、我們每 time 給 5 視角 —— vanilla 拿了**更多**視角仍崩,變因確定是監督乾淨度。

---

## 5. Poster 怎麼引用(建議措辭)

**Claim:**「在 noisy VGM 監督下,凍結 canonical + 結構化動作(SE(3)+LBS)得到 **view-consistent 的正確幾何**;joint baseline 連訓練視角都過擬合成碎片。」

- 主數字:**novel view +8.5 dB**(11.2 → 19.7),配 §3 grid 圖(挑 2–3 列)或 1 支 GIF。
- 輔助:訓練視角 digger-only +4.2 dB(誠實的 apples-to-apples);F2 物體區 3.6 dB(同 canonical 下 deform-MLP 崩 → motion model 是貢獻)。
- 防禦:sanity 25.2 dB(實作 OK)、region 表(「贏在底板?」→ appendix)。
- **不 claim** 銳利度/外觀品質;糊 = VGM 不一致顯形 → 接診斷線(故事第 3 幕)。

### 數字一致性備註
- ours 全幀 20.64(region 協議)/ 20.60(§2.5 metric 表,rotfix 重訓 `lego_v2_ctrl_rotfix`)vs 文件 20.35(舊 rotfix 標籤)—— 差異來自 checkpoint/協議,方向結論不變;**poster 建議統一引用 §2.5 的 rotfix-retrain 數字**。
- 區域 PSNR 只算 mask 內像素,數值天然比全幀低(沒有白色像素灌水),不要跨表比大小。
