# Standup(自產)實驗結果 + GT-free probe 的誠實修正(2026-06-14)

> 動機:驗證 fit-residual probe 能否在**真正無乾淨參考**的場景部署(回應「我們提出
> 一種量測 VGM 的方式」)。先排除了 jumpingjacks(自產退化、無真 orbit),standup
> 自產資料有效(真多視角、IoU 0.30)。產出兩個扎實結果 + 一個重要修正。
> 資料:`data/custom/standup_selfgen`(13 視角 × 21 幀,無 d-3dgs 乾淨參考)。

---

## 1. 前置:資料有效性篩查(jumpingjacks vs standup)

| 場景 | az180 vs input 差異 | orbit 結構 | 判定 |
|---|---|---|---|
| jumpingjacks 自產 | **最小**(5.43,全部 ~6) | 無(近正面複製) | ❌ 退化,SED 0.32 無意義 |
| standup 自產 | 大(16–23,az60 最小) | ✓ 真 front→back | ✅ 可用 |

→ **jumpingjacks 自產線應從 framing 撤掉**(SED 三場景 bar chart 要拿掉 jumpingjacks)。
standup 是真多視角(但偏乾淨,且是自定義工人 scene、非 D-NeRF standup 角色)。

## 2. 重建:ours vs vanilla(第三個場景,確認視角數效應)

全部 vs-SV4D(standup 無乾淨參考):

| 場景 | 視角數 | vanilla vs SV4D | ours vs SV4D | Δ |
|---|---:|---:|---:|---:|
| lego_v2 | 5 | 11.43(爆炸) | 20.35 | +8.9 |
| **standup** | **13** | **9.54(爆炸)** | **17.97** | **+8.43** |
| hellwarrior | 57 | 13.68 | 15.31 | +1.63 |

**method 優勢隨視角數遞減**:稀疏(5/13 視角)→ vanilla 幾何爆炸、ours(凍結 canonical)
存活 → +8.4~8.9 dB;密集(57 視角)→ vanilla 約束足夠不爆 → +1.6 dB。

![](../../runs_aux/standup_ours_vs_vanilla.png)

(每列:SV4D | vanilla | ours。**vanilla col = 教科書級黑色尖刺爆炸**,放射狀暗針 —
正是 `analyze_spike_artifact` 量到的「各向異性 p95 1727、暗針沿視線」的視覺版,
比 hellwarrior 那張更戲劇,**poster 機制圖首選**。)

## 3. ★ GT-free probe 的誠實修正(本實驗最重要產出)

standup 無乾淨參考,正是 probe「無參考部署」的真實考驗。**純 GT-free R_vgm cone 反了:**

| az | 0(input) | 60 | 120 | 180 | 240 |
|---|---:|---:|---:|---:|---:|
| R_vgm | **0.159(最高)** | 0.124 | 0.098 | 0.098 | 0.132 |

spearman(方位距離, R_vgm) = **−0.82**(可靠錐應為正:input 最可信=殘差最低)。

**核對 lego/hellwarrior 的同一個量**(純 R_vgm vs 方位距離,不用參考):
- lego −0.18、hellwarrior +0.40、standup −0.82 → **單獨 R_vgm 從來不是乾淨 cone**。

**結論:headline ρ=0.82/0.87 是 R_vgm vs araw(=|SV4D−乾淨 d-3dgs|,需要乾淨參考)的
相關,不是 standalone GT-free。** 沒有 capacity floor 減掉「fit 難度基線」時,R_vgm 量的
是「目標多銳利」(input 原始輸入最銳利、最難 fit → 殘差最高)而非「多不一致」。

→ **probe 是參考校準型儀器**:真正不可替代的價值是 **per-pixel 定位**(SED 做不到),
但需要乾淨參考校準/驗證。「GT-free」應理解為「不需逐幀乾淨 4D video,但需要乾淨參考
校準」,**不是**「無參考可部署」。**真正免參考的是 SED。**

已修正:`instruments_explained.md` §2、`academic_framing.md` C1。

## 4. 淨影響(對 framing)

不是壞消息,是更誠實清晰的分工:
- **SED** = 真正免參考的幾何儀器(無參考場景用它)。
- **probe** = 參考校準型,但能 per-pixel 定位歸因(有參考時用它的獨特價值)。
- **reference cone** = ground truth。
- 知道邊界比誇大「五儀器都 GT-free」更可信 —— 審稿人更買單。

## 5. 待辦(誠實性收尾)
- 從 `instruments_sed_fv4d.png` / instruments_table 移除或標註 jumpingjacks(退化資料)。
- standup vanilla 黑尖刺圖可併入 `spike_artifact_explanation.md` 當主視覺。
- artefacts:`outputs/custom/partrigid_standup_selfgen_ours`、`standup_selfgen_vanilla_node`、
  `runs_aux/{standup_ours_vs_vanilla,standup_canon_sanity_*,standup_video_check}.png`。
