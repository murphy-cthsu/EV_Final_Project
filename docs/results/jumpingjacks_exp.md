# Jumpingjacks 實驗總整理 — poster 素材頁

> 2026-06-14。第三個**完整**場景(proper 57-view 格式、正確相機、含 d-3dgs 乾淨參考)。
> 是第二個 articulated 場景,也是第一個**乾淨重建的 articulated**。把兩個 n=1 的論點
> (oracle gap 趨勢、probe 驗證)推到 **n=3**。資料:`/mnt/HDD_1/cthsu/jumpingjacks`。

---

## 1. 實驗設定

| 項目 | 內容 |
|---|---|
| 監督 | SV4D 9 視角單環(elev 0,az 0–330)× 21 幀,576² |
| 相機 | **原生 transforms_sv4d2_math(正確)** → 配準 IoU 0.45(全場最好) |
| 乾淨參考 | **有 d-3dgs_video**(9 視角 × 21 幀)→ 能算 vs-clean / oracle gap / validated probe |
| Canonical | d-3dgs Deformable canonical(`point_cloud/iteration_40000`,aniso p95=457) |
| 協議 | 同 lego/hellwarrior:leak-free motion-gated + rotfix,8000 iters |

與前兩個自產場景(jumpingjacks_5v 退化、standup 無參考)的關鍵差異:**這個裝備齊全**。

---

## 2. 重建:乾淨的 articulated 場景(near-lego)

| 場景 | 類型 | ours vs clean | vs SV4D | overfit gap |
|---|---|---:|---:|---:|
| lego | rigid | 20.35 | — | +6.0 |
| **jumpingjacks** | **articulated** | **21.03** | 22.66 | −1.6 |
| hellwarrior | articulated | 13.51 | 15.31 | −1.8 |

**jumpingjacks ours = 21.03 dB vs clean,跟 lego 同級** —— 第一個乾淨重建的 articulated
場景。**證明「articulated → 重建崩」不是必然**:hellwarrior 崩(13.5)是因為它的 SV4D
監督噪音大,不是因為 articulated 本身。失敗是**內容依賴**的。

3 欄 gallery(SV4D | clean d-3dgs GT | ours;身體姿勢正確,僅快速運動的手臂有 fuzz):

![](../../runs_aux/jumpingjacks_gallery.png)

動態(21 幀跳躍動作,SV4D | clean GT | ours):

![](../../runs_aux/jumpingjacks_v0.gif)

![](../../runs_aux/jumpingjacks_v4.gif)

### ours vs vanilla(誠實:乾淨監督下 vanilla 沒爆,優勢縮小)

| 方法 | vs clean | 備註 |
|---|---:|---|
| vanilla SC-GS | 18.38 | **大致完整**(非崩塌),僅頭部局部黑尖刺 + fuzz |
| **ours** | **21.03** | +2.65;凍結結構避開尖刺 |

![](../../runs_aux/jumpingjacks_ours_vs_vanilla.png)

**重要(打破純視角數趨勢)**:jumpingjacks(9 視角、**乾淨監督**)vanilla 只 +2.65、
沒爆;但 standup(13 視角、近似相機)vanilla 爆到 +8.43。→ **vanilla 是否爆炸不是
單看視角數,而是視角數 × 監督一致性 × 相機精度**。ours 的優勢 = 避開黑尖刺失效模式
(凍結結構),**這在 vanilla 會爆時最大**(lego +8.9、standup +8.4),監督乾淨時縮小
(jumpingjacks +2.65、hellwarrior +0.83)。誠實措辭:ours ≥ vanilla 全部場景,差距
隨「監督越稀疏/越不一致」放大。

---

## 3. ★ Oracle gap 三場景趨勢(把 n=1 推到 n=3)

| 場景 | 類型 | oracle(floor) | ours | **oracle gap** |
|---|---|---:|---:|---:|
| lego | rigid | 20.96 | 20.35 | **0.6 dB** |
| **jumpingjacks** | **articulated(週期/簡單)** | 24.60 | 21.03 | **3.6 dB** |
| hellwarrior | articulated(複雜) | 22.74 | 13.51 | **9.2 dB** |

**精緻化的結論(比之前 n=1 強得多):**
- articulated 確實比 rigid 受更多 supervision damage(3.6, 9.2 ≫ 0.6) ✓
- 但 articulated 的**量級內容依賴**:jumpingjacks(週期性、對稱、簡單運動)只 3.6,
  hellwarrior(多肢複雜、大 pose change)到 9.2。
- → oracle gap 不只是「rigid vs articulated」二分,而是**隨運動複雜度連續增長**的
  supervision-damage 量尺。三點成趨勢,不再是單點軼事。

左:jumpingjacks probe cone(gap=R_vgm−R_clean 追 reference,ρ=0.83);
右:oracle gap 三場景趨勢(0.6 < 3.6 < 9.2 dB)。

![](../../runs_aux/jumpingjacks_probe_and_oraclegap.png)

---

## 4. ★ Fit-residual probe:第三個驗證點(且修正 standup 的教訓)

有了乾淨參考 + floor,probe 正確重現 cone:

| 場景 | corr(gap, araw) | R_clean floor |
|---|---:|---|
| lego | ρ=0.82 | 小且平 |
| **jumpingjacks** | **ρ=0.83**(Pearson 0.82/Spearman 0.83) | range 0.0074(平) |
| hellwarrior | ρ=0.87 | 小且平 |

**關鍵:這驗證了 standup 揭露的修正**——probe 用的是 **capacity-controlled gap
(R_vgm − R_clean)** vs 參考不一致,**需要 floor**。jumpingjacks 有 floor → ρ=0.83 重現;
standup 無 floor → 純 R_vgm 反向(ρ=−0.82)。**有參考校準時 probe 可靠、且 per-pixel 定位;
無參考時退回 SED。** 三場景的 ρ=0.82/0.83/0.87 是 probe 作為「參考校準型定位儀器」的證據。

---

## 5. 對 framing 的淨貢獻

jumpingjacks 一次補強三個之前的弱點:
1. **「乾淨 articulated 重建」存在**(21 dB)→ hellwarrior 的崩是內容不是物類,framing 更準。
2. **oracle gap 趨勢 n=1 → n=3**(0.6 / 3.6 / 9.2,隨運動複雜度)→ supervision-damage 量尺成立。
3. **probe 驗證 n=2 → n=3**(ρ=0.83)+ 正面印證 standup 的「需要 floor」修正。

→ 更新 `academic_framing.md` C2(物類翻轉改成「內容依賴的 supervision-damage 量尺」)、
`hellwarrior_exp.md`(移除「articulated n=1」caveat)。

## 6. artefacts
- 模型:`outputs/custom/partrigid_jumpingjacks_{ours,floor}`、`jumpingjacks_vanilla_node`
- 資料:`data/custom/jumpingjacks{,_d3dgs_sup}`、`outputs/custom/jumpingjacks_d3dgs_ref`
- probe:`runs_aux/fit_residual_jumpingjacks.npz` + pixmap
- 修了 `build_scene_dataset.py` 缺 view_tag(已手動補進 transforms)、probe 單仰角 guard。
