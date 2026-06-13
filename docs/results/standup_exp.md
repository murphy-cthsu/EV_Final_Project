# Standup(自產)實驗總整理 — poster 素材頁

> 2026-06-14 整理。第三個場景(自產工人 scene,13 視角,**無乾淨 d-3dgs 參考**)。
> 與 lego/hellwarrior 互補:這是「稀疏視角 + 無參考」的 GT-free regime。
> 產出兩個扎實結果 + 一個重要的誠實修正。完整紀錄:`standup_selfgen_findings.md`。
> 圖:`runs_aux/standup_ours_vs_vanilla.png`、`standup_canon_sanity_*.png`。

---

## 1. 實驗設定

| 項目 | 內容 |
|---|---|
| 監督資料 | 自產 SV4D:3 elevation ring × (input + 4 novel az) = **13 視角** × 21 幀,576² |
| 來源 | `/mnt/HDD_1/cthsu/sv4d_p1_out/standup_5v_elev{0,5,10}`(P1 自產管線) |
| alpha | 非白門檻(`build_selfgen_dataset.py`) |
| 相機 | 復用 lego_v3 poses by (elev,az) tag(**近似,無原生 metadata**) |
| Canonical | `standup_scgs_default_node`(乾淨,aniso p95=17.4) |
| 評估 GT | **無 d-3dgs 乾淨參考** → 只能 vs-SV4D + GT-free 診斷 |
| 物件 | 寫實工地工人蹲→站(articulated);資料偏乾淨 |

跟 lego/hellwarrior 的關鍵差異:**稀疏視角(13 vs 57)、無乾淨參考**。

### 資料有效性篩查(對比 jumpingjacks)
| 場景 | novel-view 對 input 幅度 | orbit | 判定 |
|---|---|---|---|
| jumpingjacks 自產 | ~5–7(近正面複製) | ✗ 退化 | ❌ 棄用 |
| **standup 自產** | **~16–23(真 front→back)** | ✓ | ✅ 可用 |

canonical 對 SV4D 前景 IoU = 0.30(~hellwarrior 0.28 水準,可拟合吸收)。

---

## 2. 重建:ours vs vanilla(確認「視角數決定 method 優勢」)

全部 vs-SV4D(無乾淨參考):

| 場景 | 視角數 | vanilla | ours | Δ |
|---|---:|---:|---:|---:|
| lego_v2 | 5 | 11.43(爆炸) | 20.35 | +8.9 |
| **standup** | **13** | **9.54(爆炸)** | **17.97** | **+8.43** |
| hellwarrior | 57 | 13.68 | 15.31 | +1.63 |

**⚠️ 修正(jumpingjacks 後)**:原本想說「method 優勢隨視角稀疏度遞增」,但
jumpingjacks(9 視角、乾淨監督)vanilla 只 +2.65 沒爆,打破純視角數趨勢。
正確說法:**vanilla 是否爆炸 = 視角數 × 監督一致性 × 相機精度**。standup 爆(+8.43)
是「13 視角 + 近似相機(復用 lego_v3,無原生 metadata)」共同所致 —— 相機不準本身
就會讓監督不一致 → vanilla 爆。ours 優勢 = 避開黑尖刺失效模式,在 vanilla 會爆時最大。

![](../../runs_aux/standup_ours_vs_vanilla.png)

(每列:SV4D | vanilla | ours。**vanilla 欄 = 教科書級放射狀黑針爆炸** —— 正是
`spike_artifact_explanation.md` 量到的「暗針沿視線、各向異性 p95=1727」的視覺顯影,
比 hellwarrior 更戲劇,**poster 黑尖刺機制圖首選**。ours 凍結結構保住身體。)

動態 ours vs vanilla(SV4D | vanilla 黑針爆炸 | ours)+ novel view:

![](../../runs_aux/scene_videos/standup_selfgen_compare_v0.gif)

![](../../runs_aux/scene_videos/standup_selfgen_novel.gif)

---

## 3. ★ GT-free probe 的誠實修正(本場景最重要產出)

standup 無乾淨參考,正是 probe「無參考部署」的真實考驗。**純 GT-free R_vgm cone 反了:**

| az | 0(input) | 60 | 120 | 180 | 240 |
|---|---:|---:|---:|---:|---:|
| R_vgm=\|ours−SV4D\| | **0.159(最高)** | 0.124 | 0.098 | 0.098 | 0.132 |

spearman(方位距離, R_vgm) = **−0.82**(可靠錐應為正)。核對 lego/hellwarrior 同一個量
(純 R_vgm vs 方位距離):lego −0.18、hellwarrior +0.40、standup −0.82 →
**單獨 R_vgm 從來不是乾淨 cone**。

**結論:headline ρ=0.82/0.87 是 R_vgm vs araw(=\|SV4D−乾淨 d-3dgs\|,需要乾淨參考)
的相關,不是 standalone GT-free。** 沒有 capacity floor 減掉「fit 難度基線」時,R_vgm
量的是「目標多銳利」(input 原始輸入最銳利、最難 fit → 殘差最高)而非「多不一致」。

→ **probe 是參考校準型儀器**:真正不可替代的價值是 **per-pixel 定位**(SED 做不到),
但需乾淨參考校準/驗證。真正免參考的是 SED。已修正
`instruments_explained.md`、`academic_framing.md` C1。

---

## 4. 誠實邊界
- 全部 vs-SV4D(無乾淨參考)→ 沒有 vs-clean、oracle gap、reference cone、capacity floor。
- 相機近似復用(無原生 metadata)→ R_vgm 的 per-azimuth 受配準汙染(正是 §3 反向的部分原因)。
- standup 資料偏乾淨、且是自定義工人 scene(非 D-NeRF standup 角色)—— 來源待確認;
  vanilla 爆炸與 ours 重建是可信的(同管線),probe 反向 cone 是 GT-free 邊界的證據。

## 5. Poster 怎麼引用
- **重建線**:standup 補上「視角數 → method 優勢」趨勢的中間點(13 視角),
  且給黑尖刺機制最佳視覺。
- **診斷線**:standup 是 probe 適用邊界的負面證據 → 收進「知道邊界」的誠實論述,
  不是失敗。
- **不要 claim**:standup 有 oracle gap / 有 validated cone(都需要乾淨參考,沒有)。
