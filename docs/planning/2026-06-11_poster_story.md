# Poster 敘事定稿 — 怎麼說這個故事

> 2026-06-11。整張 poster 的故事線 + 標題 + 口頭 pitch。圖檔對應 `figs/`,版面細節見 `poster_diagnosis_panel.md`。

---

## 標題(選一)

1. **"When Reconstruction Fails, Measure the Generator: Diagnosing Multi-view Video Diffusion with a Frozen 4D Probe"** ← 推薦,故事弧全在標題裡
2. "The Cone of Reliability: Quantifying Spatio-temporal Inconsistency of Multi-view Video Generation"
3. "Structure/Motion-Decoupled 4D Gaussians as a Probe for VGM Inconsistency"

副標(小字):*Reconstructing 4D from SV4D 2.0's generated multi-view video — and turning its failure modes into a measurement instrument.*

---

## 故事弧(5 幕,= poster 的閱讀動線)

### 第 1 幕 · Setting — 「一段影片,能換到一個 4D 場景嗎?」
單目影片 → SV4D 2.0 生成多視角影片(57 視角 × 21 幀,可控相機)→ 理論上足夠重建 4D。
**但生成的視角彼此不一致** —— VGM 不保證物理正確。問題:這樣的監督能用嗎?

### 第 2 幕 · 我們的方法 — 「凍結結構,只學動作」
假設有一個乾淨的靜態 canonical 3DGS(一次掃描)。凍結它,只學 part-rigid SE(3)+LBS 動作(885K 參數)。
**結果:+8.6 dB over joint baseline**(11.43 → 20.03),同 canonical 換成 16M deform-MLP 也只有 11.89 → **贏的是 inductive bias,不是參數量**。

### 第 3 幕 · 轉折 — 「但輸出是糊的。是 bug 嗎?」(故事的 hinge)
跟 CAT4D 級的結果比,我們的 novel view 有 fuzz。查證:
- **不是 bug**:input-pose view 渲染乾淨(21.12 dB);fuzz 只出現在 VGM 不一致的地方。
- **也不是容量**:同一套結構換乾淨監督(oracle/floor),hellwarrior 直上 22.7 dB —— 差距全來自監督噪音(7 組對照排除軌跡/mask/gate/關節/canonical pose,見 hellwarrior_exp.md)。
- **是 formulation**:CAT4D joint-optimize,讓幾何「吸收」生成噪音;我們凍結結構,不一致**無處可藏**,只能顯形。
- → **頓悟:讓重建變差的特性,正是量測儀器需要的特性。** 一個固定、已知正確的參考 = calibration probe。

### 第 4 幕 · 發現 — 「VGM 有一個可靠錐」
用這個 probe + 乾淨參考,系統性刻畫 SV4D 2.0:
- **空間**:input azimuth 37.5 dB → 斜後方 19.4 dB(18 dB 落差);每 10° elevation −0.77 dB。
- **時間**:37.5% 應靜止像素假性「在動」;input view 5.7% → 偏軸 57%。
- **Pose**:運動內容偏位 lego 11→28 px、hellwarrior 36→53 px(對齊參考;azimuth 主軸)。
- **定位**:hallucination 集中在輪廓邊緣(1.4×);81% 外觀錯、19% 無中生有、0% 漏掉 → VGM 知道物體**在哪**,不知道沒看過的角度**長怎樣**。
- **物類翻轉 failure mode**:rigid → 空間崩;articulated → 時間 flicker(9.2×)。
- **Oracle gap(supervision damage,新)**:同管線同 oracle 協議,rigid 只被傷 **0.6 dB**、articulated 被傷 **9.2 dB**(20.96→20.35 vs 22.74→13.51);overfit gap 同步變號(lego +6.0 在去噪 → hellwarrior −1.8 在 fit 噪音)→ 重建端跟診斷端給出同一張失敗地圖。圖:`runs_aux/hellwarrior_gallery/orbit_novel.gif`(novel-pose 軌道,oracle | ours 並排)。

### 第 5 幕 · 儀器 — 「不需要 GT 也能量」(novel 貢獻)
**Fit-residual probe**:把物理約束模型擬合到 VGM 輸出,**擬合不了的就是不一致**。
- 完全 GT-free(`R_vgm` per view 即重現 cone:lego ρ=0.82、hellwarrior ρ=0.87)。
- Fitting 自動吸收 registration(裸靜態比對失敗處它成功)。
- **三儀器三角驗證**:reference-based cone / 我們的 fit-residual / SED 純幾何(3.9–5.9× over control,lego ρ=0.74),加 FV4D 場域標準數字(遮底板後 471/873 —— hellwarrior 比 lego 更不一致)。
- **+ oracle gap 當第四個獨立訊號**(0.6 vs 9.2 dB,只需乾淨參考訓一次 floor)—— 四路互證。

### 收尾 · Takeaways
1. VGM 監督要**按視角加權信任**(我們的 smart-photo loss 是一個實例)。
2. 下一代 VGM 該補的是 **articulated 內容的時間一致性**(那是 failure 軸)。
3. 凍結式解耦重建 = 免費的 VGM 診斷儀 —— **重建與診斷是同一個 pipeline 的兩個輸出**。

---

## 30 秒口頭 pitch(站在 poster 前講)

> 「我們想用影片生成模型當免費的多視角監督來重建 4D。我們的解耦方法比 joint baseline 好 8.6 dB —— 但輸出有點糊。我們本來以為是 bug,查證後發現:糊的地方剛好就是生成模型自己不一致的地方,因為我們凍結了結構,噪音無處可藏。所以我們反過來把它當量測儀器:量出 SV4D 有一個『可靠錐』—— 只有 input view 周圍可信,偏 120° 後保真度掉 18 dB、一半的靜止像素在假動;而且 articulated 內容傷得更重 —— 同一套管線,rigid 場景只被生成噪音傷 0.6 dB,articulated 場景傷 9.2 dB。而且我們的 fit-residual 量法完全不需要 ground truth,跟純幾何的 epipolar 檢驗、跟有 GT 的比對,三個獨立儀器互相印證。」

## 一句話版(被問 "so what?" 時)

> 「在大家都想用 VGM 當監督的時代,我們給出第一張『哪裡能信、哪裡不能信』的量化地圖,以及一個不需要 GT 的量法。」

---

## Poster 版面總圖(A0 橫式,左方法右分析)

```
┌────────────────────────── 標題 + 作者 ──────────────────────────┐
│ 左欄(方法,P1)              │ 右欄(診斷,P2)                  │
│ ① Setting + pipeline 圖      │ ④ 可靠錐 hero 圖(D1)           │
│    (slide1/slide2 drawio)    │ ⑤ 時空+定位+物類+oracle gap     │
│                              │    (D2+D3 + orbit 對比圖)       │
│ ② 重建結果表 + gallery        │ ⑥ ★fit-residual probe(D4)      │
│    (+8.6dB, 18× 少參數)      │    + 三儀器對照表                │
│ ③ 轉折框:「fuzz 不是 bug」   │ ⑦ Takeaways + limitations(D5)  │
│    v0=21.12dB 證據 ← 故事樞紐 │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

**關鍵設計:③ 轉折框放正中間偏左**,是兩欄的橋 —— 讀者從「方法不錯」走到「咦為什麼糊」再走到「原來糊是訊號」,整張 poster 的動線就是故事弧。

---

## 誠實邊界(放 limitations,口頭被追問也用這套)

- 單一 generator(SV4D 2.0)、兩個物件場景;cone 的普遍性 across generators 是 future work。
- 診斷相關性 n=9 視角 bin,0.67–0.88(中強,非完美)。
- 需要一個乾淨的靜態 canonical(pose 對齊與否已用 7 組對照排除:hellwarrior 重建差是 SV4D articulated 噪音所致,同結構乾淨監督 22.7 dB;probe 與 oracle-gap 量測皆不受影響)。
- FVD 數字是 small-N,只做相對比較。
