# Methods — Structure/Motion-Decoupled 4DGS as a VGM-Inconsistency Probe

> 詳細方法敘述,給 poster / 報告用。每個設計都標了「想說什麼」+「適合的圖」。
> 來源:`docs/design/pipeline_summary.md`、`docs/results/lego_exp.md`(component ablation)、
> `docs/results/spike_artifact_explanation.md`、`scripts/train_partrigid_hier.py`。
> 數字皆 lego_v2(SV4D 5 view × 21 frame),GT = 獨立 Deformable-3DGS 訓在乾淨 D-NeRF lego。

---

## 0. 一頁總結(敘事主線)

我們的目標是「**從一張圖 → 用 video generative model(SV4D)生多視角影片 → 重建 4D 物件**」。

1. **問題**:直接拿 vanilla SC-GS(自由 16M deform-MLP)去 fit SV4D 生出來的影片 → 因為
   SV4D 各視角**彼此不一致(hallucination)**,自由變形場把噪音**吸進幾何**,長出暗的、
   各向異性的**針狀高斯(black spikes)**,物體區崩掉(digger 區只剩 3.59 dB)。
2. **我們的設計**:不讓變形場碰幾何 —— **凍結一個乾淨 canonical 當「結構」,只學一個
   大幅受限的「動作」參數化(part-rigid SE(3) + LBS)**,並用 **noise-robust 的監督**
   (motion-gated smart-photo + silhouette + ARAP + trajectory)。結果:11.43 → 20.64 dB,
   針狀 artifact 在架構上長不出來,失敗變成**有界、形狀守恆**。
3. **轉折(同一管線的第二個 output)**:即使機制到位,off-axis 仍模糊 —— 因為 SV4D 本質
   self-inconsistent,沒有任何單一連貫的 4D 能同時滿足矛盾的視角。我們把這個**有界殘差**
   反過來當**量測**:per-view fit-residual = VGM 不一致度,重現 reliability cone
   (lego ρ=0.82 / hellwarrior 0.87 / jumpingjacks 0.83)。
   → **重建與診斷是同一管線的兩個輸出。**

一句話:**把結構與動作解耦,讓動作模組「藏不住噪音」;藏不住的噪音就成了可量測的不一致。**

---

## 1. 問題:為什麼 vanilla SC-GS 在生成影片上會壞

- **監督資料本身矛盾**:SV4D 是 per-view 各自 hallucinate 的;同一個 3D 點在不同視角的
  顏色/位置對不上。單一連貫 3D 模型物理上無法同時滿足。
- **自由變形場會把矛盾吸進幾何**:deform-MLP(~16M 未知數)為了壓低 photometric loss,
  學出**沿視線拉長、暗、高不透明度**的針狀高斯來「各視角各說各話」。
  - 量化:各向異性 p95 **vanilla = 1727 vs 乾淨 canonical = 64(27×)**;中位數都 ~5
    → artifact 在分布尾巴,不是整體。
- **後果(region-decomposed PSNR,訓練視角)**:

  | 模型 | full | digger(7.6% px) | baseplate(9.5%) | background(82.9%) |
  |---|---|---|---|---|
  | Vanilla SC-GS | 11.43 | 10.66 | 5.27 | 13.64 |
  | F1 warm-start(給 canonical 不凍) | 11.55 | 10.87 | 5.42 | 13.70 |
  | F2 frozen + deform-MLP | 11.89 | **3.59** | 4.81 | **62.91** |
  | **Ours** | **20.64** | **14.90** | **14.70** | **24.80** |

  - F2 的「+0.46 贏 vanilla」是**假象**:物體區 3.59 dB(MLP diverge,物體渲不出來),
    全幀分數全靠白背景 62.9 dB 撐。→ **同一顆凍結 canonical,MLP 物體直接崩,我們 14.9 dB。**
  - **贏在 inductive bias 不是容量**:同 canonical 換 16M deform-MLP = 11.89,我們 885K = 20.64。

**想說的**:失敗不是「沒 fit 好」,而是「自由度太高 → 噪音洩漏進幾何」。
**適合的圖**:vanilla 黑針 render vs ours 乾淨 render(並排);各向異性分布直方圖(1727 vs 64)。

---

## 2. 核心想法與設計原則

| 原則 | 內容 | 為什麼 |
|---|---|---|
| **結構 / 動作解耦** | 結構來自乾淨 canonical(凍結,`requires_grad=False`),只學動作 | VGM 噪音集中在**結構**(跨視角不一致);單視角內**動作**相對乾淨(已量:per-view spread 3.5 dB、時間軸平) |
| **不用 raw-RGB photometric** | 動作只靠 noise-robust 訊號監督 | photometric 無法分辨「對的動作配錯的像素」vs「錯的動作」,梯度照樣把 artifact 灌進 deform |
| **大幅壓縮動作搜尋空間** | per-part SE(3) 取代 per-Gaussian deform-MLP | 16M 未知數 → 數百;噪音可洩漏的維度等比例變小,也是黑針長不出來的根因 |
| **多源弱監督** | silhouette / trajectory / ARAP / smoothness 共識 | 各訊號朝不同方向失敗,共識更穩 |

**想說的**:這四條原則合起來 = 「讓動作模組沒有空間去解釋 SV4D 的矛盾」。
**適合的圖**:一張「自由度對比」概念圖(deform-MLP 16M 雲狀箭頭 vs part-rigid 100×SE(3) 離散箭頭)。

---

## 3. Pipeline(兩條輸入 → 預處理 → 訓練 → 輸出)

### 3.1 輸入(只有兩個合法輸入,訓練從不碰 clean GT)
- **凍結乾淨 canonical 3DGS**:~114K Gaussians,`requires_grad=False`,input pose vs 真 GT ≈ **34 dB**。
- **SV4D 多視角影片**:V views × T frames(lego 5 視角 × 21 幀,576²),noisy / hallucinated。
- 附:相機 c2w(Blender convention)、FOV。

### 3.2 預處理(Stage A′–D,**零可學參數**)
- **Stage A′ — canonical re-render**:把凍結 canonical 在各視角靜態渲染 → `I_canon(v)`,
  當 smart-photo 的乾淨參考。
- **Stage B — motion mask `m_v`**:per-pixel 時間變異 + Otsu ∩ 前景 → 哪些像素在動。
- **Stage C — per-Gaussian part label**:把 Gaussian 投影到所有視角,用 `m_v` 投票
  → {arm(會動), body(靜), unassigned}。**只有 arm 之後會動。**
- **Stage D — 3D part trajectory**:DLT 三角化各視角 motion-mask 形心 → `path(t) ∈ ℝ^{T×3}`
  (給 SE(3) 平移當 init / trajectory loss 目標)。

### 3.3 訓練(Stage E,**唯一會學的**;885K 參數,~4 min / 1 GPU)
- **arm 高斯 K-means 成 K = 100 cluster**;每 cluster 每 frame 一個剛體 **SE(3)** `(R_{k,t}, t_{k,t})`。
- **LBS skinning**:每個 arm 高斯綁 **K_lbs = 6** 個最近 cluster,Gaussian-kernel 權重
  `raw_w = exp(−d²/2σ²)` 正規化 → 軟混合,不撕裂。body 高斯權重 ~0 → 跟著凍結結構不動(灰底)。
- **per-(cluster,time) 3D scale residual**:修 canonical 形狀掃動時的 streaking。
- **per-Gaussian XYZ residual**:小、有正則的連續非剛性微修(剛體 SE(3) 之上的修正)。

變形公式(per arm 高斯 `x`,frame `t`):
```
x'(t) = Σ_k  w_k(x) · [ R_{k,t} (x − c_k) + c_k + t_{k,t} ]   +  Δx_res(x,t)
        (1 − Σ_k w_k) · x   ← body / 邊界:留在凍結 canonical
```

### 3.4 輸出(同一管線兩個 output)
- **重建**:形變後 4DGS `G(t)` —— input pose 乾淨、off-axis fuzzy。
- **診斷**:per-view fit-residual = VGM 不一致量測。

**想說的**:預處理全是「免費」的(無學習),所有訓練參數都在那個受限的動作模組。
**適合的圖**:pipeline 主圖(已有);Stage B/C 視覺(motion mask + part label,已有);
LBS skinning 權重熱圖(可新做)。

---

## 4. Loss 設計

```
L = λ_silh  · L_silhouette(SAM-2 mask)
  + λ_smart · L_smart-photo              ← motion-gated, leak-free(關鍵創新)
  + λ_ARAP  · L_ARAP(cross-cluster rigid)
  + λ_smooth· ‖Δ_t(R, t)‖²               ← SE(3) 時間平滑
  + λ_traj  · ‖mean(cluster pos) − path(t)‖²
```

### Smart-photo(leak-free, motion-gated)— 最值得展開的 loss
- **靜態像素** `m_v = 0`:`w(p) = exp(−α · |I_sv4d(p) − I_canon(p)|)`,α = 16
  → canonical 定義正解;SV4D 偏離乾淨參考 = hallucination → 權重→0,**過濾掉**。
- **運動像素** `m_v = 1`:`w(p) = 1`
  → canonical 不適用(那裡本來就要動),信 SV4D,交給 silhouette / ARAP / trajectory。
- **leak-free**:所有參考只從兩個合法輸入導出,訓練**從不碰 clean GT**
  (舊版拿 clean Deformable-3DGS render 當參考 = 同 eval GT → soft leak;已改掉,代價 −0.37 dB)。

**想說的**:我們不是無腦 photometric;是「只在我們相信的地方用 RGB」。
**適合的圖**:smart-photo 權重圖 —— 一張影格疊上 `w(p)` 熱圖(亮=信任,暗=被濾掉的 hallucination),
旁邊放 motion mask 顯示運動區 w=1。**這張最能展示「我們設計了什麼」。**

---

## 5. 哪些是 load-bearing(誠實標註)

| 元件 | 地位 | 證據 |
|---|---|---|
| **凍結 canonical + part-rigid SE(3)+LBS** | ★ **核心** | +8.6 dB 全靠它;16M deform-MLP 同 canonical 只 11.89 → 贏 inductive bias 非容量 |
| **凍結 canonical(單獨)** | 最大單一貢獻 | vanilla 11.43 → 15.91 靜態(**+4.5**) |
| smart-photo(leak-free) | 有用但**非主因** | leak fix 只 −0.37;最大 loss 機制 **+1.42** |
| K-sweep(K=10→100) | 甜蜜點 | **+0.69**;K=200/300 反而過擬合噪音 |
| per-Gaussian XYZ residual | motion 模組內最大 | **+0.44** |
| per-(cluster,time) scale | 小修 | **+0.22** |
| Stage D trajectory init/loss | **可選** | lego/hellwarrior 拔掉無損 |
| ARAP / smoothness | 穩定性輔助 | — |
| ~~per-Gaussian rotation residual~~ | **拒絕** | **+0.04 → 誠實負結果** |
| `--d_rot_zero` | 協議修正(必帶) | 加性旋轉殘差打亂朝向 ~0.3 dB;訓練+eval 都要帶 |

**想說的**:故事是 **decoupling(結構/動作)**,不是某個花俏的 loss;而且我們誠實報負結果。
**適合的圖**:component contribution bar(已做 `method_M2`);model comparison bar(已做 `method_M1`)。

---

## 6. 從重建到診斷(轉折)

- 「凍結結構 + 受限動作」讓**動作模組藏不住噪音** —— 殘差只能出現在 VGM 不一致處。
- **fit-residual(per-view)= 不一致量測**:
  - **空間**:per-pixel residual 熱圖 → 不一致長在哪(邊界/幻覺區)。
  - **角度**:per-view PSNR 隨方位角掉 → reliability **cone**;殘差 gap 與原始不一致
    **Spearman ρ = 0.82**(lego)/ 0.87(hellwarrior)/ 0.83(jj)→ 量到的就是不一致本身。
- 對照 vanilla:自由 deform-MLP 把噪音吸進全域幾何 → 黑針(各向異性 1727);
  ours 形狀守恆,失敗從「無界幾何崩塌」變「有界、可歸因的不一致顯影」。

**想說的(soft 版)**:同一個受限 probe,免費多給一個 output —— 一張不一致地圖。
**適合的圖**:residual 熱圖(已修好 `block5_inconsistency_heatmap`)+ cone(已有 `block5..._cone`)。

---

## 7. 設計部件貢獻表(可直接做圖)

| # | Component | 做什麼 | 貢獻(lego_v2) |
|---|---|---|---|
| A | 凍結乾淨 canonical | 結構不可學,噪音無法吸進幾何 | **+4.5**(11.43→15.91 靜態) |
| B | Motion mask + part 投票 | 只讓動的部位動,限制 DOF | 限制自由度(防 body 吸噪) |
| C | K=100 cluster SE(3)+LBS | per-part 剛體動作 + 軟混合 | K-sweep **+0.69** |
| D | Smart-photo(VGM 信心加權) | 濾掉 hallucination 像素 | **+1.42** |
| E | per-(cluster,time) scale | 修掃動 streaking | +0.22 |
| F | per-Gaussian XYZ residual | 連續非剛性微修 | +0.44 |
| G | Stage D 軌跡 init | 質心三角化當 SE(3) 平移 init | 可選 |
| — | per-Gaussian rot residual | 試過每高斯旋轉 | +0.04 → **拒絕** |

---

## 8. 圖規劃(你決定要做哪些)

> 標記:✅ 已有 · 🟡 有素材可重做 · 🔴 要新做

### A. 展示「我們設計了什麼」(目前最缺)
1. 🔴 **Smart-photo 權重圖**:一張 SV4D 影格 + `w(p)=exp(−α|SV4D−canon|)` 熱圖疊圖,
   標出靜態區「被濾掉的 hallucination」vs 運動區「w=1 信任」。**最能講清楚 leak-free 設計。**
2. 🔴 **自由度對比概念圖**:deform-MLP(16M,雲狀)vs part-rigid(100×SE(3),離散箭頭)
   —— 「我們把搜尋空間從 16M 壓到數百」。
3. 🟡 **LBS skinning 權重熱圖**:單一 cluster 的 `w_k(x)` 在高斯上的軟分布(no tearing)。
4. ✅ **Stage B/C 視覺**:motion mask + part label(`block6_motion_mask` / `block6_part_assignment`)。

### B. 展示「為什麼這樣設計 work」(對照/消融)
5. 🔴 **黑針對照**:vanilla(黑針)vs ours(乾淨)render 並排 + 各向異性直方圖(p95 1727 vs 64)。
   素材:`runs_aux/recon_inout/*`(我已渲過 vanilla 帶針 / pruned 乾淨)、`spike_artifact_analysis.png`。
6. ✅ **model comparison bar**:Vanilla / 16M-MLP / Ours(`method_M1`)。
7. ✅ **component contribution bar**:各部件 +dB(`method_M2`)。
8. 🟡 **canonical ablation 4-model**:clean GT | vanilla | F1 不凍 | F2 凍+MLP | ours
   (`runs_aux/novel_view_compare/canonical_ablation_4model.png`)。

### C. 展示「診斷」(第二個 output)
9. ✅ **residual 熱圖**(`block5_inconsistency_heatmap`)+ **cone ρ=0.82**(`block5..._cone`)。
10. 🟡 **跨物件 generality**:cone 在 lego/hellwarrior/jj 都成立(`vgm_cone_generality.png`)。

**我的建議優先序(若要新做)**:#1(smart-photo 權重圖)> #5(黑針對照+直方圖)> #2(自由度概念圖)
> #3(LBS 權重)。前兩個直接補上「設計可見化」這個目前最大的缺口。
