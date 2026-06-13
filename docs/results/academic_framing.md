# 怎麼學術地包裝這個 project(2026-06-13)

> 不是「更好的 4D 重建」論文(我們輸 CAT4D、articulated 上甚至沒贏 vanilla)。
> 是「用重建當儀器去量測生成式影片模型」的 **measurement / probing** 研究。
> 這個 genre 本身是受認可的(類比:「生成模型到底懂不懂 3D?」這類分析論文)。

---

## 0. 一句話 thesis(整個 project 收斂到這一句)

> **凍結一個已知正確的幾何先驗(乾淨靜態 canonical + 物理約束動作),擬合到生成式
> 影片模型產出的多視角影片 —— 因為結構被凍結、不一致無法被吸收進幾何,VGM 的
> 時空不一致就被「逼出來」成為可量測、可定位、且不需 GT 的訊號。**

所有觀察都是這一句的推論:
- 可靠錐 / flicker / oracle gap = 被逼出來的不一致,量出來了。
- 黑色尖刺 = 你「不凍結」時會發生的事(vanilla 把不一致吸進針狀幾何);凍結 → 改成局部顯影。
- fit-residual probe = 同一件事的 GT-free 讀數。

---

## 1. Positioning:這是什麼類型的貢獻

| 不是 | 是 |
|---|---|
| SOTA 4D 重建 | 用重建當**校準探針**去量測 VGM |
| 新的生成模型 | 對既有 generator(SV4D 2.0)的**系統性失效刻畫** |
| 大規模 benchmark | 一個**方法論 + 概念驗證**(1 generator、2 場景,誠實標明) |

**Genre**:probing / diagnostic / measurement study。貢獻在「**怎麼量**」和「**量到什麼**」,
不在「重建分數」。

---

## 2. Contribution claims(按可辯護性排序,附證據)

**C1 ★ 方法論(最novel):Reconstruction-as-measurement。**
凍結結構 → 不一致無處可藏 → 殘差即不一致。實作成 fit-residual probe,
其 per-view 殘差與 reference 不一致量相關(lego ρ=0.82、hellwarrior ρ=0.87),
**per-pixel 可定位**(其他儀器做不到),並與幾何(SED)、分佈(FV4D)、
supervision-damage(oracle gap)互相印證。
> 關鍵論點:「freeze 讓重建變差」與「freeze 讓量測變準」是**同一個性質**。
> ⚠️ **誠實邊界(standup 實驗,2026-06-14)**:那個 ρ 是 **vs 乾淨參考(araw)**
> 的相關,**不是** standalone GT-free。純 R_vgm vs 方位距離在無參考場景(standup)
> 反向(ρ=−0.82,被 fit 難度汙染)。所以 probe 是**參考校準型**儀器(價值在
> per-pixel 定位,需參考驗證),**不是**「無參考可部署」。免參考的是 SED。
> 這個邊界本身要寫進論文 —— 知道邊界比誇大覆蓋更可信。

**C2 經驗發現:SV4D 2.0 的失效結構。**
(a) 空間可靠錐(input 37.5 → 偏軸 19.4 dB,−0.77 dB/10° elev);
(b) 物類相依失效:rigid → 空間崩;articulated → 時間 flicker 主導;
(c) **oracle gap = supervision damage,隨運動複雜度連續增長**(n=3):
rigid 0.6 dB(lego)< articulated 週期 3.6 dB(jumpingjacks)< articulated 複雜 9.2 dB
(hellwarrior)—— articulated 受更多傷,但量級內容依賴(不是物類二分)。

**C3 機制解釋:黑色尖刺 = 多視角不一致的幾何形體。**
同物體控制實驗:noisy 監督下高斯各向異性 p95=1727(乾淨 64),且 needle 只在 noisy
變暗 2×、沿視線排列。**SED 用數字量的、尖刺用形狀畫的,是同一件事。**

**C4 方法(誠實 scoped):structure/motion 解耦 4D-GS。**
監督允許時(rigid/乾淨)近 oracle 重建(lego +8.9 dB,20.35 vs ceiling 20.96);
監督夠髒時失敗模式**有界可解釋**(vs vanilla 無界針狀崩塌,p95 64 vs 1727)。
> C4 不是「我們贏」,是「我們證明探針在乾淨情況校準正確 + 失敗時仍受控」。

---

## 3. 敘事弧(poster / 短文都用這個)

1. **Setup**:VGM 當免費多視角監督來重建 4D —— 誘人但沒人驗證過監督可不可信。
2. **Attempt**:解耦凍結 canonical + part-rigid 動作。rigid(lego)成功,+8.9 dB 近 oracle。
3. **Obstruction(轉折)**:articulated(hellwarrior)**沒有任何方法救得回**
   (ours +0.83 over vanilla,兩者都 ≪ oracle 22.7);到處黑色尖刺。
4. **Insight**:障礙來自 generator 不是 method;而**「凍結結構」正是讓障礙可量測的關鍵**
   —— 藏不進幾何的不一致只能變成殘差。黑色尖刺 = 不一致的幾何顯影。
5. **Contribution**:把探針變成多儀器診斷,刻畫 SV4D 的可靠錐 / 物類翻轉 / oracle gap,
   交叉驗證、部分 GT-free。
6. **Scope + future**:1 generator、2 場景;cross-generator 是 future work。

> 這是經典的「negative result 轉 positive」:重建失敗 → 量測成功。**重建分數不是
> headline,探針校準(rigid 成立)+ 失效刻畫(articulated)才是。**

---

## 4. 建議標題 + 摘要骨架

**標題**:*When Reconstruction Fails, Measure the Generator: A Frozen 4D Probe for
Multi-View Video Diffusion Inconsistency*

**摘要骨架(5 句)**:
1. VGM 被當作免費多視角監督,但它們的時空不一致從未被系統量化。
2. 我們凍結一個乾淨靜態 canonical、只學物理約束動作,把它擬合到 SV4D 2.0 的生成影片。
3. 因為結構不能吸收不一致,擬合殘差**就是** per-view / per-pixel 的不一致量(GT-free)。
4. 它重現了一個獨立乾淨參考測得的「可靠錐」(ρ 0.82–0.87),並與對極幾何、FVD、
   oracle-gap 四種獨立儀器一致;據此我們刻畫 SV4D 的可靠錐、物類相依失效、
   與一個量化「監督損傷」的 oracle gap。
5. 我們進一步證明常見的黑色尖刺 artifact 是多視角不一致的幾何形體,而凍結結構把它
   從全域崩塌轉成局部、可歸因的顯影。

---

## 5. 要 claim / 不要 claim(避免被一眼看穿)

| ✅ 要說 | ❌ 不要說 |
|---|---|
| 探針在 rigid 上校準正確(near-oracle) | 「我們的 4D 重建 SOTA / 比 CAT4D 好」 |
| 殘差 = 不一致,GT-free,可定位 | 「我們在 hellwarrior 重建比較好」(+0.83 撐不住) |
| 五儀器交叉驗證同一張失效地圖 | 「這是 VGM 通用 benchmark」(只有 1 generator) |
| 凍結讓失敗有界可解釋(p95 64 vs 1727) | 「黑尖刺是我們修好的 bug」(它是訊號不是 bug) |
| oracle gap 量 supervision damage | 過度宣稱相關係數(n=9,ρ 0.67–0.88,說「中強」) |

---

## 6. 誠實 limitations(主動寫,反而加分)
- 單一 generator(SV4D 2.0);cross-generator 是 future work。
- 3 個完整場景(lego rigid + jumpingjacks/hellwarrior articulated);oracle gap
  趨勢 n=3、probe 驗證 n=3。cross-generator 仍是 future work。
- 診斷相關 n=9 視角 bin,ρ 0.67–0.88(中強)。
- 需要乾淨靜態 canonical(一次掃描);probe 對它的品質有依賴。

---

## 7. 最划算的補強(若要再強一點,按 CP 值)
1. ✅ **已完成(jumpingjacks)**:oracle gap 趨勢 n=3、probe 驗證 n=3。第三個 articulated
   (trex/mutant)可選,進一步密集化趨勢。
2. **bouncingballs**(rigid 平移)→ oracle-gap 趨勢第三點(平移<旋轉<articulated),且可能給乾淨 demo。
3. **per-part fit-residual** → 「哪條肢體最不一致」,診斷粒度升級(便宜)。
4. cross-generator → 最強但最貴,明確留 future。

---

## 8. 一段話總結(被問「所以你們貢獻是什麼」)

> 「在大家想把影片生成模型當免費 3D 監督的時代,我們給出**第一個用『凍結幾何探針』
> 把 VGM 多視角不一致變成可量測、可定位、GT-free 訊號的方法**,並用它畫出 SV4D 2.0
> 的『哪裡能信、哪裡不能信』地圖 —— 還順帶證明了大家常見的黑色尖刺 artifact,其實
> 就是這個不一致的幾何形體。重建只是探針;失效本身才是我們量到的東西。」
