# Standup 實驗總整理 — poster 素材頁

> 2026-06-14。**兩個 standup**:(A) proper benchmark(原生 9 視角 + 乾淨 d-3dgs 參考,
> 有 oracle gap)= 主結果;(B) 自產 13 視角(無參考)= GT-free probe 邊界發現。
> 圖:`standup_proper_ours_vs_vanilla.png`(A)、`standup_ours_vs_vanilla.png`(B)。

---

# Part A — Proper standup(原生資料,有乾淨參考)★ 主結果

## A1. 實驗設定

| 項目 | 內容 |
|---|---|
| 監督資料 | 原生 SV4D 2.0,**9 視角** × 21 幀,576²(`/mnt/HDD_1/cthsu/standup/sv4d2/standup_r20`) |
| 評估 GT | **有 d-3dgs 乾淨參考**(189 renders = 9×21)→ 能算 vs-clean / oracle gap / floor |
| Canonical | `standup_scgs_default_node`(乾淨,28,804 高斯) |
| 物件 | 工地工人 蹲→站(articulated) |

跟 jumpingjacks 一樣是「原生 9 視角 + 乾淨參考」的完整 benchmark;與 hellwarrior(57 視角)、
lego(5 視角)構成視角數光譜。

## A2. 重建:vanilla 爆炸 vs ours(全 vs clean d-3dgs)

| 方法 | vs clean | 備註 |
|---|---:|---|
| vanilla SC-GS(joint,16M) | **6.63** | 幾何爆炸(放射狀黑針) |
| **ours**(凍結 canon + part-rigid) | **16.71** | +10.09 over vanilla |
| oracle / floor(訓乾淨 GT) | **24.74** | |

**oracle gap = 24.74 − 16.71 = 8.02 dB**;**ours − vanilla = +10.09 dB**(lego 之後最大)。

![](../../runs_aux/standup_proper_ours_vs_vanilla.png)

(每列:SV4D | clean GT | vanilla | ours。**vanilla 欄 = 教科書級放射狀黑針爆炸**,
比 hellwarrior 更戲劇 —— `spike_artifact_explanation.md` 量到的「暗針沿視線、
各向異性 p95=1727」的最佳視覺顯影,**poster 黑尖刺機制圖首選**。ours 凍結結構保住工人身體。)

## A3. ★ Oracle gap 四場景趨勢(n=4,3 個 articulated)

| 場景 | 類型 | oracle(floor) | ours | **oracle gap** |
|---|---|---:|---:|---:|
| lego | rigid | 20.96 | 20.35 | **0.6** |
| jumpingjacks | articulated(週期) | 24.x | 21.03 | **3.6** |
| **standup** | **articulated(蹲→站)** | **24.74** | **16.71** | **8.02** |
| hellwarrior | articulated(複雜多肢) | 22.74 | 13.51 | **9.2** |

**0.6 < 3.6 < 8.0 < 9.2** —— oracle gap(= supervision damage)**隨運動複雜度連續增長**,
不是 rigid/articulated 二分。standup 把趨勢從 n=3 加密到 n=4,3 個 articulated 點
證明這是趨勢而非單點。**這是 C2 最強的經驗發現。**

---

# Part B — 自產 standup(13 視角,無參考)= GT-free probe 邊界

> 自產 SV4D(`sv4d_p1_out/standup_5v_elev{0,5,10}`,13 視角,復用 lego_v3 近似相機,
> **無乾淨參考**)。價值:probe「無參考部署」的真實考驗。完整:`standup_selfgen_findings.md`。

## B1. ★ GT-free probe 的誠實修正(本場景最重要產出)

standup 自產無乾淨參考。**純 GT-free R_vgm cone 反了:**

| az | 0(input) | 60 | 120 | 180 | 240 |
|---|---:|---:|---:|---:|---:|
| R_vgm=\|ours−SV4D\| | **0.159(最高)** | 0.124 | 0.098 | 0.098 | 0.132 |

spearman(方位距離, R_vgm) = **−0.82**(可靠錐應為正)。核對三場景純 R_vgm vs 方位:
lego −0.18、hellwarrior +0.40、standup −0.82 → **單獨 R_vgm 從來不是乾淨 cone**。

**結論:headline ρ=0.82/0.87 是 R_vgm vs araw(=\|SV4D−乾淨 d-3dgs\|,需乾淨參考)的相關,
不是 standalone GT-free。** 沒有 capacity floor 減掉「fit 難度基線」時,R_vgm 量的是
「目標多銳利」(input 原始輸入最銳利、最難 fit → 殘差最高)而非「多不一致」。

→ **probe 是參考校準型儀器**:不可替代的價值是 **per-pixel 定位**(SED 做不到),
但需乾淨參考校準/驗證。真正免參考的是 SED。已修正 `instruments_explained.md`、`academic_framing.md` C1。

## B2. 自產重建(vs-SV4D,無參考)
| 場景 | 視角 | vanilla | ours | Δ |
|---|---:|---:|---:|---:|
| standup 自產 | 13 | 9.54(爆炸) | 17.97 | +8.43 |

(都 vs-SV4D;近似相機 → 監督不一致 → vanilla 爆。與 Part A 的原生 9 視角結論一致:
vanilla 爆炸 = 視角數 × 監督一致性 × 相機精度。)

---

## 誠實邊界
- Part A 原生資料、有乾淨參考 → oracle gap / floor / vanilla 對照都可信。
- Part B 自產、近似相機、無參考 → 只有 vs-SV4D;probe 反向 cone 是 GT-free 邊界的證據(不是失敗)。

## Poster 怎麼引用
- **重建線**:standup proper = vanilla 爆炸 +10.09 dB 的乾淨對照 + 黑尖刺機制最佳視覺。
- **診斷線**:oracle gap 8.0 把趨勢推到 n=4(0.6<3.6<8.0<9.2);自產 standup 是 probe 邊界的誠實證據。
- **要 claim**:proper standup 有 oracle gap 8.0(有乾淨參考)。
- **不要 claim**:自產 standup 有 validated cone(無參考,probe 反向)。
