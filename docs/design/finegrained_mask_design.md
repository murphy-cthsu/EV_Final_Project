# Fine-grained mask 的架構/loss 設計空間（2026-06-13）

> 背景:hellwarrior 階梯實驗(L1/L1b/L2 全平)排除了「軌跡 init」;binary gate
> 量化顯示 57% 高斯被壓到 ~26% 活動力(median arm_weight 0.263);root cause
> 候選 = canonical pose 不對齊 + binary gate。本文件列出 fine-grained mask
> (L2 temporal-profile parts / SAM 3D-prompt parts)能用在哪。
> 判定實驗:floor eval(容量上限)、allmove(gate 全開)→ 結果見 §5。

## 0. 手上的 mask 資產(粒度由粗到細)

| 資產 | 粒度 | 性質 |
|---|---|---|
| Stage B motion mask | binary(moving/static) | intensity variance,邊緣偏置,低紋理區漏 |
| L2 temporal-profile labels | P 個 motion part,per-view per-pixel + per-Gaussian | timing-based,跨視角對應免費,邊界粗 |
| SAM 3D-prompt masks(原型) | P 個 part,dense 邊界 | 身分由 3D 保證,邊界準,t>0 需 warp |

## 1. 架構層

### A3 ★ Canonical re-pose(Stage A″,優先 —— 直接打 root cause)
用 t=0 的 per-part mask(57 視角)對凍結 canonical 做**一次性 per-part SE(3) 對齊**:
渲染 part p 的高斯 alpha vs 該視角 SAM part mask,對 per-part SE(3)(+global)做
silhouette fitting。蹲姿 canonical → 站姿 t=0,之後 Stage E 只需學 t 間的動作。
- 為什麼是 mask 的正確用法:whole-object silhouette 對「左右腿交換」「蹲/站」
  幾乎不變(剪影對稱),**per-part mask 才有把肢體拉到正確位置的梯度**。
- 純前處理,不動訓練程式;失敗也不污染現有結果。
- 風險:SAM part mask 在偏軸視角品質掉(用 input ring ± 2 視角就夠)。

### A1 Part-scoped 模型結構(取代 binary gate)
- per-Gaussian gate 從 scalar arm_weight 改成 **part 隸屬**(gaussian_motion_part / SAM 投票)。
- K-means **在 part 內**做(每 part K_p ≈ K/P),LBS 鄰居**限定同 part**
  → 殺掉跨肢 bleeding(現在 K_lbs=6 鄰居可跨兩條腿,關節處 candy-wrap)。
- **body 也是一個 part**(低 K、自己的 SE(3)),不再是凍結的背景
  → 全身動作(蹲下)不被 gate 壓死。allmove 實驗 = 這個方向的 lower bound
  (allmove 連 part 結構都沒有,只開 gate)。

### A2 兩層 skeleton(part SE(3) ∘ sub-cluster SE(3))
part-level 剛體(從 part mask DLT/質心估)+ cluster 殘差。比 A1 多一層
歸納偏置,但工程量大;deadline 內不建議,寫進 future work。

## 2. Loss 層

### L-A ★ Part-aware ARAP(一行級改動,先做)
現在 ARAP 對「空間相鄰的 cluster」一視同仁 → **把關節焊死**(跨 part 的
cluster 對也被拉成剛性)。有 per-cluster part 標籤後:同 part 全強度,
跨 part 降權(×0.1 或 0)。讓關節能彎,肢體內保持剛性。

### L-B Part-silhouette loss(最強的新監督,×P render 成本)
對每個 part p:只渲染 part p 高斯的 alpha,跟該視角的 SAM part mask 比。
- whole-silhouette 的致命盲點:它無法區分「哪條腿蓋住哪個區域」——
  左右腿換位剪影不變。part-silhouette 直接監督**對應關係**。
- 成本:每 iter ×P 次 render → 每 iter 隨機抽 1 個 part 即可(期望等價)。
- 需要 t>0 的 part mask:SAM video predictor 沿時間傳播(per view 一次),
  或 L2 temporal-profile labels 當粗版(已有,免費)。

### L-C Per-part smart-photo gating(把診斷接回重建)
smart-photo 的 motion-gating 目前 binary。改成 per-part 信心:
part p 在視角 v 的權重 ∝ 該 part 的 fit-residual / flicker 統計
(診斷輸出直接變 loss 權重)→「重建與診斷是同一 pipeline」的故事閉環。

### L-D 2D–3D part 一致性 loss(輕量)
投影高斯到視角,若 2D 像素 part 標籤(SAM/L2)≠ 該高斯 3D part 標籤,
罰 silhouette-style 距離。比 L-B 便宜(不用多次 render),監督力較弱。

## 3. 診斷線的延伸(免費的 poster 加分)
per-part fit-residual:probe 按 part 分解 → 「articulated 內容裡**哪條肢體**
最不一致」。SV4D 的 failure 軸是 articulated 時間 flicker(9.2×),part 級
定位是現有三儀器都做不到的,且只是把現有 residual 按 gauss_part 分組。

## 4. 建議順序(deadline 內)
1. **L-A part-aware ARAP**(~30 min,搭 L2 labels 立即可跑)
2. **A3 canonical re-pose**(~半天,打 root cause;SAM 原型已驗證)
3. **A1 part-scoped LBS + body-as-part**(~半天,等 allmove 判定)
4. L-B part-silhouette(timebox 半天,需 t>0 mask 傳播)
5. L-C/診斷延伸(poster 加分,各 ~2h)

## 5. 判定實驗結果

| 實驗 | 設定 | PSNR vs clean |
|---|---|---:|
| ours(SV4D 監督,binary gate) | ctrl_rotfix | 13.51 |
| **floor(乾淨 d-3dgs 監督,同 gate 同 canonical)** | `hellwarrior_cleancanon_floor` | **22.74** |
| allmove(SV4D 監督,gate 全開) | `hellwarrior_cleancanon_allmove` | 13.38 |
| part-ARAP(SV4D 監督,joint 鬆綁 ×0.1) | `hellwarrior_cleancanon_partarap` | 13.53 |

**判定:floor = 22.74 推翻了所有「模型結構」假說 —— 同 gate、同蹲姿 canonical,
乾淨監督下動作 fit 得很好。hellwarrior 的瓶頸是 SV4D articulated 監督噪音本身。**
- oracle gap:lego 0.6 dB(20.96−20.35)vs hellwarrior **9.2 dB**(22.74−13.51)
  → 「同一管線,rigid 內容只損失 0.6 dB,articulated 內容損失 9.2 dB」——
  與診斷線(articulated temporal flicker 9.2×、FV4D 873 vs 471)互相印證。
- **oracle-gap 本身 = 第四個診斷儀器**(supervision-noise damage,per-scene scalar)。
- 含義:§1/§2 的架構/loss 改動對 hellwarrior 重建的預期收益有限
  (assignment 不是 binding constraint);它們的價值移到 (a) 學習式 part 結構
  取代手工 Stage B/C(故事更乾淨),(b) per-part 診斷粒度,(c) 噪音魯棒的
  loss(L-C per-part 信心 gating)才是對症下藥。

## 6. MoE 化:routed rigid experts(回應「用 MoE MLP 學各種 motion」)

**核心觀察:我們的模型已經是一個「手工版 MoE」** ——
LBS 權重 = gating network(但目前是固定的:空間 kernel × binary arm gate),
K=100 個 per-time SE(3) = experts(但 expert 指派由 k-means 固定)。
MoE 化 = 把 routing 變可學習。對應表:

| MoE 概念 | 我們現有 | MoE 化後 |
|---|---|---|
| expert | per-cluster SE(3)(t) 表 | 同(保持 rigid!) |
| gating | 固定 spatial kernel × arm_weight | **per-Gaussian 可學 logits,top-k softmax** |
| 靜態/動態二分 | binary gate(w>0.5) | **identity expert**(凍結零變換,路由到它=靜止) |
| 部位數 | binary(+L2 的 P=4 只用於 init) | E 個 expert,軟性、可學 |
| load balancing | 無 | entropy + kNN 圖平滑(鄰近高斯路由相似) |

**為什麼 expert 必須是 rigid SE(3) 而不是 MLP**:兩個既有證據 ——
F2(16M deform-MLP 在同 canonical 下 11.89 vs 我們 885K 的 20.35:自由度會吸噪音)
+ floor(rigid experts 容量足以 fit 乾淨 hellwarrior 到 22.74:不缺容量)。
**MoE-of-MLPs 會把 F2 的失敗模式請回來;MoE-of-rigid-SE(3) 才是對的版本。**

**v1 實作草圖**(`train_partrigid_hier.py`,~1-2h):
- `route_logits = nn.Parameter(N, E)`,init = log(現有 lbs_weights + ε)(warm start,
  只會比現狀好);`w = softmax(topk(logits, 6))`取代固定 buffer。
- expert 0 = identity(trans/aa 凍結為 0),取代 binary gate。
- 正則:entropy(每顆高斯承諾少數 expert)+ kNN graph smoothness(logits);
  router lr 低(1e-3)防止吸噪音。
- 預期:重建小幅改善(assignment 非 binding constraint);主要產出是
  **學到的 routing = part discovery**(可視化 = poster 圖)+ per-expert fit-residual
  (診斷粒度:哪個 motion component 最不一致)。

**誠實預期**:在 SV4D 噪音監督下,可學 router 有吸噪音風險(正則要夠強);
依 floor 判定,它不會解掉 hellwarrior 的 9.2 dB gap —— 那需要 L-C 類的
噪音魯棒 loss(per-part 信心 gating)或更乾淨的 generator。
