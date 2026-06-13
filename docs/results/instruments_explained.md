# Probe vs SED vs Reference cone — 各是什麼、量什麼、為什麼需要(2026-06-14)

> 三個儀器都在量「SV4D 多視角不一致」,差別在**需要知道多少**(資訊需求)
> 與**看得到什麼**(量測豐富度)。它們構成一條光譜,不是重複。

---

## 一張表看懂

| | 需要什麼 | 量到什麼 | 定位/歸因 | per-azimuth cone | 失效盲點 |
|---|---|---|---|---|---|
| **SED**(對極幾何) | 只要圖 + **相對**相機 | 純**幾何/pose** 不一致 | ✗(成對標量) | 弱(ρ=0.74,偏軸 SIFT 變少) | 看不到外觀/時間;需紋理 |
| **Fit-residual probe**(★ours) | 乾淨 canonical + **正確相機** | **任何**物理解釋不了的(空間+時間+外觀) | ✓ per-pixel,可歸因 | 強(ρ=0.82/0.87,GT-free) | 殘差混因(需 capacity floor 對照);需 canonical |
| **Reference cone**(D-series) | **完整乾淨 4D 參考**(d-3dgs video) | 全部,直接 | ✓ | 是(定義 cone 的基準) | 需要的正是我們想免掉的東西 |

---

## 1. SED:需要最少,看得最窄

**機制**:兩個生成視角間找 SIFT 匹配點 → 若 SV4D 3D 一致,匹配點必須落在
**相機幾何**決定的對極線上 → 偏離 = 同一物理點被生成到不一致位置。

- **需要**:生成圖 + 相對相機。不要 3D 模型、不要 canonical、不要乾淨參考。
  → 對極只看相對關係,**絕對相機錯了也能跑**(自產 jumpingjacks 能用就是這原因)。
- **量到**:純幾何 / pose 不一致。SV4D 3.9–5.9× over 乾淨 control;自產 4 場景 0.32–0.48 px(floor 0.16)。
- **看不到**:外觀幻覺、顏色、時間 flicker;需紋理(SIFT)→ 偏軸模糊匹配變少 →
  SED 反而可能下降 → per-azimuth **不是乾淨 cone**。無法 per-pixel 定位、無法分因。

## 2. Fit-residual probe:需要更多,看得最寬(ours)

**機制**:把「乾淨 canonical + 低自由度物理動作」擬合到 SV4D 影片;擬合不掉的
殘差 R_vgm = |our render − SV4D| = 任何物理 4D 解釋不了的部分 = 不一致。

- **需要**:乾淨靜態 canonical(一次掃描)+ **正確相機**(canonical 必須投影到物體上)。
  拟合能吸收**中等**錯位(hellwarrior IoU 0.28 可用),但不能吸收完全錯的相機帧
  (jumpingjacks IoU 0.19 失敗 = 復用了錯坐標系的相機)。
- **量到**:空間 + 時間 flicker + 外觀,全折進 photometric 殘差(比純幾何寬)。
  **per-pixel 可定位、可歸因**。GT-free 重現 cone:lego ρ=0.82、hellwarrior ρ=0.87。
  capacity floor 對照:乾淨擬合殘差小 3.6–4.8×、平坦 7–9% → per-view 變化是不一致非容量。
- **弱點**:殘差混因(需 floor 對照);需 canonical + 相機。

## 3. Reference cone(D-series):需要最多,當驗證基準

SV4D vs 乾淨 d-3dgs 逐 (view,t) 比對。看得最全、最直接,但需要**完整乾淨 4D 重建**
—— 正是我們想免掉的東西。所以它的角色是**驗證**(probe/SED 跟它對得上),不是部署。

---

## 4. 為什麼需要好幾個 —— 兩個獨立的理由

### 理由 A:獨立失效模式 → 交叉驗證 = 結論穩健
三者的**假設與盲點互相獨立**:SED 假設有紋理、只看幾何;probe 假設有 canonical、
混所有因;reference 假設有乾淨 4D。**如果這些假設完全不同的方法都指向同一個 cone,
那 cone 就不是任何單一方法的 artifact** —— 它是 SV4D 真的性質。這就是「三角驗證」的力量:
- probe(ρ=0.82/0.87)+ SED(ρ=0.74,幾何)+ FV4D(分佈)+ oracle gap(0.6 vs 9.2)
  + reference cone,五個獨立讀數同向。

### 理由 B:覆蓋不同的「資訊可得性」情境 → 一套工具箱
真實部署時,你手上有什麼不一定:
- 有乾淨 4D 參考 → 用 reference cone(最全)。
- 有乾淨 canonical + 正確相機 → 用 **probe**(GT-free 且可定位)= 甜蜜點。
- 只有圖 + 相對相機(連 canonical 都沒) → 用 **SED**(假設最少)。

→ **它們不是冗餘,是覆蓋光譜。** probe 在「不要乾淨 4D 參考、但要 canonical+相機」這格,
SED 在「什麼模型都不要」這格。jumpingjacks 的教訓正好證明這個分工:probe 因相機帧
錯而失效的地方,SED 仍能跑(免相機)——**這正是為什麼要兩個都有**。

---

## 5. 一句話總結(被問「probe 跟 SED 差在哪、為什麼都要」)

> 「SED 假設最少(只要圖 + 相對相機)但只看幾何、不能定位;我們的 probe 多要一個
> 乾淨 canonical,換來能看空間+時間+外觀、且 per-pixel 定位歸因。兩者失效模式獨立,
> 互相印證讓 cone 結論穩健;而且覆蓋不同的資訊情境 —— 有 canonical 用 probe,
> 沒有就退回 SED。重建失敗的 jumpingjacks 上 probe 因相機失效、SED 仍可跑,
> 就是這個分工的證據。」
