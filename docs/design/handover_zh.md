# 專案交接文件（中文版）— Structure/Motion 解耦 4D-GS 與 VGM 不一致性診斷

> 更新於 2026-06-01。對象：要 sync 進度的隊友。先從頭讀一遍,再用 codebase map + 3 天計畫。
> **最新策略狀態都在 `meetTW_checkpoint_0601/`**(deck.md、framework.md、autonomous_session_summary.md)。這份是穩定的總覽。

---

## 0. TL;DR（先讀這段）

- **課程 final project → 交付物是一張 POSTER**(不是投頂會)。Scale/contribution 不用大。約 **3 天、2 人**。
- 我們先做了一個**方法**:從影片生成模型(SV4D 2.0)產出的 noisy 多視角影片,重建出乾淨的 4D 高斯場景 —— 做法是**凍結一個乾淨的靜態 canonical 3DGS,只學動作**(part-rigid SE(3) + LBS)。結果:在同樣 noisy 監督下比 vanilla SC-GS baseline **高 +8.6 dB**。
- PI feedback 後我們**轉了 framing**:與其跟 CAT4D 拚重建品質(會輸 —— 他們 generator 更好且 joint optimize),不如把這個解耦架構當成**診斷 VGM 哪裡、以何種形式時空不一致的儀器**。
- **Poster 上要放的兩個乾淨結果:**
  1. **重建**(方法成立):解耦 + part-rigid 動作 = +8.6 dB,參數比 deform-MLP baseline 少 18×。
  2. **診斷**(轉向):SV4D 有個 **「可靠錐 reliability cone」** —— 在 input view 附近保真度高,偏離視角後(空間 + 時間)都退化;而我們的 **fit-residual probe** 用自己的模型當儀器、**完全不需 GT**,在兩個場景都重現這個 cone。

---

## 1. 專案在做什麼

**Setting。** 輸入 = 一段單目影片。一個現成的影片生成模型(**SV4D 2.0**)把它轉成 **object-centric 多視角影片**(可控 elevation/azimuth 的新視角;我們用 57-view grid = 7 elevation × 9 azimuth,21 frames)。這些生成的視角**空間/時間不一致**(VGM 不保證物理正確)。我們另外假設有一個 **乾淨的靜態 canonical 3DGS**(t=0);實驗中它來自乾淨 D-NeRF/Deformable-3DGS 預訓,實際部署時就是一次快速靜態掃描。

**兩條線:**
- **(A) 重建方法** —— 凍結 canonical、只學動作 → 乾淨、可渲染、保留 identity 的 4D。
- **(B) 診斷(目前主 framing)** —— 用解耦架構去**量測** VGM 的不一致:在哪裡(視角、影像區域)、以何種形式(空間 / 時間 / pose / 幾何)。

**為什麼轉向。** 凍結 canonical 在重建上輸(結構吸收不了生成噪音 → 輸出模糊),但這正是**量測儀器**需要的(一個固定、已知正確的參考)。**讓重建變差的特性,正是讓我們成為有效探針的特性。** 完整論證:`meetTW_checkpoint_0601/formulation_justification.md`、`cat4d_comparison.md`。

---

## 2. 方法 — 重建 pipeline(線 A）

給定凍結 canonical + SV4D 多視角影片,**5 階段** pipeline。A–D 是**零學習參數的前處理**,只有 Stage E 在學。

| 階段 | 做什麼 | 輸出 |
|---|---|---|
| **A. 凍結 canonical** | 一個乾淨靜態 3DGS(約 114k 高斯),所有屬性 `requires_grad=False` | 乾淨、抗噪的結構 |
| **B. Motion mask** | 每視角逐像素**時間變異** + Otsu 門檻 → 哪些像素在動 | `m_v` masks |
| **C. Part assignment** | 把 canonical 高斯投影到所有視角,用 `m_v` **多視角投票** → {arm, body, unassigned} | per-Gaussian part id |
| **D. Arm trajectory** | 用 DLT 三角化各視角 motion-mask 質心 → 3D 動作軌跡(Stage E 的 init) | `(T,3)` 軌跡 |
| **E. Motion module（學習）** | arm 高斯做 K=100 K-means;每群每時刻 **SE(3)**(旋轉+平移);**LBS** 混合最近 K_lbs=6 群;per-time per-cluster 3D scale;per-Gaussian XYZ residual。約 885K–2.1M 參數 | 變形後 3DGS G(t) |

**Loss(Stage E):** `silhouette(SAM-2) + smart-photo + ARAP(群剛性) + 時間平滑 + 軌跡-init`。

**Smart-photometric loss**(VGM 監督的關鍵技巧):per-pixel 信心權重
`w = exp(-α·|I_sv4d − I_ref|)`,再 `L = Σ|I_pred − I_sv4d|·w·alpha_mask / Σw`。
- SV4D 跟乾淨參考不一致的像素 → 權重 ~0(濾掉 hallucination)。α=16。
- **重要(leakage 歷史):** 原本 `I_ref` 用 d-3dgs render = eval GT → soft leakage。**已修**成 **motion-gated canonical-static render**(Stage A' = 把凍結 canonical 渲到每個視角;靜止區跟它比,動態區權重=1)。Leak-free headline = **20.03 dB**(原 leaky 20.40)。一律報 leak-free。

**關鍵設計理由(放 poster):**
- 凍結 vs joint:vanilla joint SC-GS = 11.43 dB(幾何爆炸);我們凍結 = 20.03。
- 結構化 SE(3)+LBS vs 自由 deform-MLP:同 canonical 換動作模組 → SC-GS 16M deform-MLP = 11.89 dB vs 我們 885K = 20.03 → **+8.14 dB;binding constraint 是 inductive bias 不是 DOF。**

---

## 3. 方法 — 診斷(線 B,目前主軸)

我們對 SV4D 的不一致做量化(對乾淨參考 d-3dgs),而且關鍵是**用我們自己的方法當儀器**。

### 3a. 描述性診斷(D-series)— SV4D vs 乾淨 d-3dgs
- **D5 空間保真度 vs 視角** → **「可靠錐」**:PSNR(SV4D, clean) 在 input azimuth = 37.5 dB → 遠側 ~19 dB(18 dB 落差);每 10° elevation −0.77 dB。
- **D3 時間 flicker** → 37.5% 應靜止的像素在 SV4D 裡假性「在動」;input view 5.7% → 偏軸高達 57%。同一個方位錐結構。
- **D6 pose drift** → 生成物體 centroid 偏離要求幾何 +0.57 px/°(對應 SV4D 2.0 論文自己寫的 failure mode)。
- **D2/D7 定位** → hallucination 在輪廓邊緣強 1.4×;誤差 81%「外觀錯」、19%「無中生有」、0%「漏掉」。
- **Generality** → cone 在 hellwarrior(articulated)也複製 —— rigid 場景空間崩、articulated 場景時間 flicker 主導(9.2×)。

### 3b. ★ Novel 量法 — capacity-controlled fit-residual probe（已驗證）
**核心想法:** 一個物理上自洽的多視角影片,能被「乾淨靜態 canonical + 低自由度物理動作」解釋。**VGM 不一致 = 任何物理 4D 場景都解釋不了的部分。** 把我們的 part-rigid 模型擬合到 VGM 視角;per-view 擬合殘差高 = 物理不一致。
- **完全 GT-free:** `R_vgm = |我們的render − SV4D|` per view,**完全不用乾淨參考**就重現 cone(lego_v3 Spearman 0.82,hellwarrior 0.87)。
- **Capacity-controlled:** 可額外擬合乾淨參考(`R_clean`,跨視角均勻)來確認 per-view 變化是不一致而非模型容量。floor 只需算一次,不是每次量測都要。
- **繞開 registration:** fitting 吸收全域錯位(裸的靜態-canonical 比對在這裡失敗 —— lego_v3 silhouette IoU 只有 0.28)。
- **定位:** 跟 MEt3R(成對靜態、reference-free)、FV4D(aggregate scalar)互補 —— 我們是 *joint 多視角 + 時間 + 物理動作* 一致性,可定位、可歸因。

---

## 4. Codebase map（怎麼跑）

環境:conda env **`scgs`**(`/home/cthsu/miniconda3/envs/scgs/bin/python`)。SC-GS 在 `third_party/SC-GS`。
**⚠️ GPU 陷阱:** 這台共用機的 **GPU 1 被別的 user 佔**(render 變 17s/frame,GPU 0/2 只要 0.01s)。**一律用 `CUDA_VISIBLE_DEVICES=0` 或 `2`。** 並設 `OMP_NUM_THREADS=6`(24 核機,numpy 會 oversubscribe)。**不要 `pkill -f <腳本名>`** —— 會殺到自己正在跑的 job。

### 核心腳本
| 腳本 | 用途 |
|---|---|
| `scripts/train_partrigid_hier.py` | **Stage E 訓練**(方法本體)。旗標:`--canon_ply --part_dir --scene_dir --v5_render_dir --k_arm 100 --lbs_K 6 --lam_photo_smart 3 --photo_smart_alpha 16 --use_per_time_scale --use_xyz_residual --use_test_too --iterations 8000`。加 `--motion_gated_smart_photo` 走 leak-free。 |
| `scripts/motion_parts_generic.py` | **Stage B+C+D**,任意 dataset:`--dataset --canon_ply`。寫到 `runs_aux/part_assignment_<ds>/`。 |
| `scripts/eval_lego_v2_hier.py` | 評估訓好的模型:`--label --scene --canon_ply [--save_renders]`。印 vs SV4D 跟 vs d-3dgs 的 PSNR。 |
| `scripts/fit_residual_probe.py` | **診斷 probe（泛化、GPU-native）**:`--scene --canon --label_vgm --label_floor --tag`。輸出 azimuth+elevation cone、corr(gap,raw)、per-pixel 圖 → `runs_aux/fit_residual_<tag>.npz`。 |
| `scripts/diagnose_vgm_inconsistency.py` | D5（空間 cone）+ D6（pose drift），SV4D vs d-3dgs。 |
| `scripts/diagnose_vgm_temporal_flicker.py` | D3（靜止區時間 flicker）。 |
| `scripts/diagnose_vgm_hallucination.py` | D2（邊緣 vs 內部）+ D7（誤差類型），`--scene`。 |
| `scripts/diagnose_cone_generality.py` | cone 跨場景複製。 |
| `scripts/render_canonical_57x21.py` | 把 canonical 渲到 57-view grid（smart-photo 參考 / 靜態比對）。 |
| `scripts/train_scgs_deform_frozen_canon.py` | F2 fairness baseline（SC-GS deform-MLP + 我們的凍結 canonical）。 |

### 重建端到端（單一場景）
```bash
# 1. 前處理 (B+C+D)
python scripts/motion_parts_generic.py --dataset lego_v3 --canon_ply outputs/custom/lego_v2_canonical/point_cloud/iteration_0/point_cloud.ply
# 2. 訓練 (GPU 0)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 python scripts/train_partrigid_hier.py --label lego_v3_A1 \
  --canon_ply <canon.ply> --part_dir runs_aux/part_assignment_lego_v3 --scene_dir data/custom/lego_v3 \
  --v5_render_dir outputs/custom/lego_v3_d3dgs_ref/renders --motion_gated_smart_photo \
  --use_test_too --k_arm 100 --lbs_K 6 --lam_arap 1 --lam_photo_smart 3 --photo_smart_alpha 16 \
  --use_per_time_scale --use_xyz_residual --iterations 8000
# 3. 評估
CUDA_VISIBLE_DEVICES=0 python scripts/eval_lego_v2_hier.py --label lego_v3_A1 --scene lego_v3 --canon_ply <canon.ply>
```

### 診斷 probe（單一場景）
```bash
# 需要一個 fit 到 SV4D 的模型 (label_vgm) 跟一個 fit 到乾淨 d-3dgs 的 (label_floor;訓練時 scene_dir=<ds>_d3dgs_sup)
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 python scripts/fit_residual_probe.py \
  --scene lego_v3 --canon <canon.ply> --label_vgm lego_v3_A1 --label_floor lego_v3_d3dgs_floor --tag legov3
```
> 註:GT-free 版本其實只需 `label_vgm`(R_vgm 自己就重現 cone);`label_floor` 只用來驗證 capacity 均勻。

---

## 5. 硬碟上的資料與模型

**Datasets**(`data/custom/`):`lego_v2`(5 視角)、`lego_v3`(57 視角)、`hellwarrior`(57 視角),+ `*_d3dgs_sup` 變體(d-3dgs render 當監督,用來訓 floor 模型)。
**乾淨 GT 參考**(`outputs/custom/*_d3dgs_ref/renders/`):獨立 Deformable-3DGS(訓在乾淨資料)的逐 (view,time) render —— 用於 eval + 診斷參考。lego_v2/v3 + hellwarrior。
**訓好的模型**(`outputs/custom/partrigid_*`):
- `partrigid_lego_v2_A1_leakfree_B` — 主重建結果(20.03 dB)。
- `partrigid_lego_v3_A1`、`partrigid_lego_v3_d3dgs_floor` — 診斷 probe(SV4D-fit + clean-floor)。
- `partrigid_hellwarrior_cleancanon_A1`、`partrigid_hellwarrior_cleancanon_floor` — hellwarrior probe。
- canonical:`outputs/custom/lego_v2_canonical/`(lego 用);hellwarrior 乾淨 canonical 在 `/mnt/HDD_1/cthsu/EV_Final_Project/outputs/hellwarrior_scgs_default_node/`。
**診斷結果**:`runs_aux/fit_residual_{legov3,hellwarrior}.npz`(+ pixmap)。
**圖 + 所有策略文件**:`meetTW_checkpoint_0601/figs/` 與 `meetTW_checkpoint_0601/*.md`。

---

## 6. 目前結果（poster 用數字）

**重建（lego_v2,vs 乾淨 d-3dgs GT）:**
| 方法 | PSNR | DOF |
|---|---:|---:|
| Vanilla SC-GS(joint,16M deform-MLP) | 11.43 | 16M |
| SC-GS deform-MLP + 我們的凍結 canonical（F2） | 11.89 | 16M |
| **我們（凍結 canon + part-rigid,leak-free）** | **20.03** | 885K |
| Oracle 天花板（我們訓在乾淨 GT 上） | 20.96 | 885K |
| 我們，僅 digger 區（排除底板） | 21.12 | — |

**診斷（SV4D 2.0）:**
- 可靠錐:37.5 dB（input azimuth）→ 19.4 dB（遠側）;每 10° elevation −0.77 dB。
- 時間 flicker:37.5% 靜止像素假性在動;input view 5.7% → 偏軸 57%。
- fit-residual probe（我們的儀器,GT-free）重現 cone:lego_v3 ρ=0.82,hellwarrior ρ=0.87。

---

## 7. 已完成 vs 還差什麼

**已完成:**
- 重建方法 + 完整 ablation（解耦、fairness F1/F2、ablation panel、ceiling、底板 artefact）。
- 修了 leak（smart-photo → motion-gated canonical）。
- 診斷 D2/D3/D5/D6/D7 + generality（lego_v3 + hellwarrior）。
- **驗證了 GT-free fit-residual 量法 probe**。
- 所有圖 + PI deck（`meetTW_checkpoint_0601/deck.md`,7 slides）。

**還差（讓 poster 更完整）:**
1. **標準 metrics**:重建比較加 LPIPS/SSIM(部分有)+ 最好加 **DreamSim**,並報 **vs-SV4D 跟 vs-clean**(「overfit gap」)。單靠 PSNR 不可靠(獎勵模糊)。
2. **MEt3R baseline**(optional,加分):reference-free 多視角一致性指標(CVPR'25),放在我們 fit-residual 旁當對照。需裝 DUSt3R(裝到 `/mnt/HDD_1`,別裝 `/home`,它 99% 滿)。
3. **fit-residual probe 的時間軸**(optional):目前是空間;加 per-frame-Δ 殘差顯示它也 track flicker。
4. **Poster 組裝**:method 圖(用 `slide2_architecture.drawio`)、重建比較(gallery GIF / 關鍵幀)、診斷 cone 圖(`fit_residual_generality.png`、`vgm_inconsistency_curves.png`)。
5. **鎖 narrative**:poster = 「Structure/Motion 解耦 4D-GS,當作探針來診斷 VGM 不一致」。重建是方法,診斷是分析。

**已知 caveat(poster 上誠實寫):**
- 診斷相關係數是每場景 n=9 點,中到強(0.67–0.88）。
- 乾淨 canonical 假設;hellwarrior 的 canonical pose 不對齊(那邊重建差,但 probe 仍有效因為 fitting 吸收掉)。
- 單一 generator(SV4D 2.0);cross-generator 是 future work(`meetTW_checkpoint_0601/benchmark_plan.md`)。

---

## 8. 3 天、2 人計畫（poster scope）

> 分工:**P1 = 重建/方法線**、**P2 = 診斷線**。Day 3 兩人一起做 poster。保持精簡 —— 這是課程 poster。

### Day 1 — 鎖結果 + 補 metric 缺口
- **P1**:在 `eval_lego_v2_hier.py` 加 LPIPS + SSIM + DreamSim;產出最終重建表(vanilla vs 我們 vs ceiling)**含 overfit gap**(vs-SV4D vs vs-clean)。重生 gallery / 關鍵幀圖。（GPU 0）
- **P2**:確認診斷圖是最終版(cone、flicker、fit-residual generality)。幫 hellwarrior 也做 per-pixel 定位圖。寫診斷那半的 poster 文字。（GPU 2）
- **兩人 EOD**:敲定 poster outline(章節 + 要哪些圖)。

### Day 2 — 各做一個「完整性」實驗 + 草擬 poster
- **P1**:擇一:(a) 用 DreamSim/LPIPS 框成「我們去噪(overfit gap 小),vanilla 過擬合(gap 大)」;(b) 乾淨地再跑一個重建場景。完成 method + ablation 圖。
- **P2**:擇一:(a) fit-residual probe 的時間軸;(b) MEt3R baseline(若 setup 順利,timebox 半天 —— DUSt3R 裝不起來就放棄)。完成診斷圖。
- **兩人 EOD**:第一版完整 poster 草稿(圖都放好,caption 寫好)。

### Day 3 — 組裝 + 潤飾
- 上午:整合、收緊文字、把 method 示意圖弄乾淨(`slide2_architecture.drawio` → 用 app.diagrams.net 匯出 PNG,機器沒 CLI)。
- 下午:內部 review、修數字/caption、印刷檢查。

### Poster 結構（建議）
1. **Problem/Setting** — VGM 給 noisy 多視角影片;我們重建乾淨 4D 並診斷 VGM。
2. **Method** — 5-stage 解耦 pipeline 示意圖。
3. **Reconstruction result** — 表(+8.6 dB,18× 少參數)+ gallery 比較。
4. **Diagnosis** — 可靠錐(空間 + 時間)+ fit-residual probe(我們的 GT-free 儀器）。
5. **Takeaways + limitations**。

### 不要做（scope 守則）
- 不要追 cross-generator benchmarking(3 天太大)。
- 不要跟 CAT4D 拚重建品質(打錯仗)。
- 不要加新架構功能 —— 方法已凍結,這是 writing+packaging 衝刺。

---

## 附錄 — 上一個自主 session 做了什麼（2026-06-01 夜間,整合自 autonomous_session_summary.md）

那一段把 fit-residual 量法從「lego_v3 初步驗證」推到「跨物類 + GT-free 完整驗證」:
1. 把 probe 泛化成 `scripts/fit_residual_probe.py`(吃參數、GPU-native、azimuth+elevation cone + per-pixel 圖)。
2. 建 `data/custom/hellwarrior_d3dgs_sup` + 訓 `partrigid_hellwarrior_cleancanon_floor`。
3. 在 lego_v3 跟 hellwarrior 都跑 probe → 兩者都重現 cone(lego ρ=0.82,hellwarrior ρ=0.87,R_clean floor 均勻)。
4. **GT-free 確認**:`R_vgm` 自己(只擬合 SV4D,不碰乾淨參考)就重現 cone(相關係數跟有減 floor 的 gap 幾乎一樣)→ clean floor 只需算一次驗證 capacity,不是每次量測都要。
5. 圖:`figs/fit_residual_{cone,generality,pixmap_legov3}.png`。文件全更新(framework Part 7、diagnosis_findings、deck Slide 7)。

**過程教訓(已寫進上面 §4):** 卡很久是因為 (a) deformation 一開始在 CPU numpy 對全部 92k 高斯算 dense LBS(1.7s/frame)→ 搬 GPU 變 0.03s;(b) GPU 1 被佔。GPU 一直有正常被呼叫,不是 GPU bug。
