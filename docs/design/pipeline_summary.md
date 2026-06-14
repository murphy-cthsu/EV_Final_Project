# Pipeline 設計總覽 — Structure / Motion-Decoupled 4DGS

> 單一真理來源，整理我們 pipeline 的所有設計選擇。2026-06-14。
> 來源：`docs/design/motion_design.md`（4 claims + stage 細節）、
> `meetTW_checkpoint_0601/slide2_description.md`（leak-free 方法）、
> `docs/design/scgs_hook_design.md`（patch 點）、`project_hellwarrior_next_benchmark`
> 與 `docs/results/spike_artifact_explanation.md`（哪些 load-bearing / 黑針機制）。

**一句話**：把「結構」和「動作」解耦 —— 結構用乾淨 canonical 凍結，動作用大幅
受限的參數化從 noisy VGM 影片學，且**不用 raw-RGB photometric loss** 把結構噪音
灌進來。

---

## 一、核心設計原則

| 原則 | 內容 | 為什麼 |
|---|---|---|
| **結構/動作解耦** | 結構來自乾淨 canonical（凍結），只學動作 | VGM 噪音集中在**結構**（跨視角不一致），單視角內**動作**相對乾淨（C1，已量測：per-view PSNR spread 3.5 dB、時間軸平）|
| **不用 raw-RGB photometric loss** | 動作只靠 noise-robust 訊號監督 | photometric loss 無法分辨「對的動作配錯的像素」vs「錯的動作」，梯度照樣把 artifact 灌進 deform（C2）|
| **大幅壓縮動作搜尋空間** | per-part SE(3) 取代 per-Gaussian deform-MLP | 16M 未知數 → 數百，噪音可洩漏的維度等比例變小（C3）；也是黑針長不出來的根因 |
| **多源弱監督** | silhouette / trajectory / ARAP / smoothness 共識 | 各訊號朝不同方向失敗，共識更穩（C4）|

> C1 有實測支持；C2 是假設並驗證；C3/C4 是以推理辯護的設計選擇。

---

## 二、Pipeline 元件（兩條輸入 → 預處理 → 訓練 → 輸出）

### 輸入（只有兩個合法輸入）
- **凍結乾淨 canonical 3DGS**：~114K Gaussians，`requires_grad=False`，input pose vs 真 GT ≈ 34 dB。
- **SV4D 多視角影片**：V views × T frames（lego 5–57 視角 × 21 幀），noisy / hallucinated。
- 附帶：相機 c2w（Blender convention）、FOV。

### 預處理（零可學參數）
- **Stage A′** — canonical 在各視角靜態 re-render → `I_canon(v)`，當 smart-photo 的乾淨參考。
- **Stage B** — motion mask `m_v`：per-pixel 時間變異 + Otsu ∩ 前景。
- **Stage C** — per-Gaussian part label：把 Gaussian 投影到所有視角，用 `m_v` 投票 → {arm, body, unassigned}。
- **Stage D** — 3D part trajectory：DLT 三角化各視角 motion-mask 形心 → `path(t) ∈ ℝ^{T×3}`。

### 訓練（Stage E，唯一會學的；885K 參數，~4 min / 1 GPU）
- K = 100 cluster **SE(3)**：per-cluster per-frame (R_{k,t}, t_{k,t})。
- **LBS** skinning：每個 Gaussian 綁 K_lbs = 6 個最近 cluster。
- per-time per-cluster 3D scale residual。
- per-Gaussian XYZ residual（小、有 regularize）。

### 輸出
- 形變後 4DGS G(t)：input pose 乾淨、off-axis fuzzy（重建輸出）。
- per-view fit-residual = VGM 不一致量測（診斷輸出）。**同一管線兩個輸出。**

---

## 三、Loss 設計

```
L = λ_silh  · L_silhouette(SAM-2)
  + λ_smart · L_smart-photo          ← motion-gated, leak-free（關鍵創新）
  + λ_ARAP  · L_ARAP(cluster-rigid)
  + λ_smooth· ‖Δ(R, t)‖²
  + λ_traj  · ‖mean(cluster pos) − path(t)‖²
```

### Smart-photo（leak-free, motion-gated）
- **靜態像素** `m_v = 0`：`w(p) = exp(−α · |I_sv4d(p) − I_canon(p)|)`，α = 16
  → canonical 定義正解，SV4D 偏離 = hallucination → 過濾掉。
- **運動像素** `m_v = 1`：`w(p) = 1`
  → canonical 不適用，信 SV4D，交給 silhouette / ARAP / trajectory loss。
- **所有參考都只從兩個合法輸入導出，訓練從不碰 clean GT。**
  （舊版用 clean Deformable-3DGS render 當參考 = 同 eval GT → soft leak；已改掉，代價 −0.37 dB。）

### SC-GS 整合點（patch `train_gui.py`，pinned SHA `3a9d2ad4`）
A 時間編碼 gating · B photometric gating（換成 masked smart-photo）· C 加 rest-state L2 ·
E cross-view gating；ARAP 改 per-edge 權重需 monkey-patch `ARAPDeformer.cal_arap_error`。

---

## 四、誠實標註：load-bearing vs 可有可無

| 元件 | 地位 | 證據 |
|---|---|---|
| **凍結 canonical + part-rigid SE(3)+LBS** | ★ **核心** | +8.6 dB 全靠它（11.43→20.03）；16M deform-MLP 同 canonical 只 11.89 → 贏 inductive bias 非容量 |
| smart-photo（leak-free） | 有用但**非主因** | leak fix 只 −0.37 dB；故事是 decoupling 不是 smart-photo |
| **Stage D trajectory init / traj loss** | **可選** | 拿掉 lego 20.79 ≥ 20.35；hellwarrior traj off / Stage D removed / 4-part / gate 全開 全平 |
| ARAP / smoothness | 穩定性輔助 | — |
| `--d_rot_zero` | **協議修正（必帶）** | 之前誤傳 d_rotation=(−1,0,0,0) 進 SC-GS 加性旋轉殘差打亂朝向（~0.3 dB）；訓練 + eval 都要帶 |

---

## 五、和診斷線的扣合

「凍結結構 + 受限動作」不只是重建方法 —— 它讓**動作模組藏不住噪音**，殘差只能
出現在 VGM 不一致處（fit-residual 重現 cone，lego ρ=0.82 / hellwarrior ρ=0.87）。

對照 vanilla SC-GS：自由 deform-MLP 把噪音吸進全域幾何 → 學出**暗、高不透明度、
沿視線拉長的針狀高斯**（各向異性 p95 = 1727 vs 凍結 canonical 64，27× 差）= 黑色
尖刺。ours 凍結結構 → 形狀守恆，物理上長不出針，失敗從「無界幾何崩塌」變成
「有界、可歸因的不一致顯影」。詳見 `docs/results/spike_artifact_explanation.md`。

→ **重建與診斷是同一管線的兩個輸出。**
