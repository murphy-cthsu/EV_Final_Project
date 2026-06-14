# Region-decomposed PSNR — 「贏在底板?」質疑的定量回答

> 2026-06-12。動機:vanilla SC-GS 的 offset view 有幾何爆炸且**沒有底板**
> (SV4D 監督沒有底板、SAM-2 mask 也把它去掉),而我們的底板是從凍結 canonical
> 免費繼承的。全幀 PSNR 的 +8.6 dB 會不會主要是「有底板」造成的?
>
> 腳本:`scripts/eval_region_psnr.py` → `runs_aux/region_psnr_lego_v2.json`。

## 協議

- lego_v2,105 (view,time) 幀,vs 乾淨 d-3dgs GT,per-frame PSNR 取平均(同主 eval 協議)。
- 區域(同一組 mask 套用到所有模型):
  - **digger** = SV4D SAM-2 alpha(被監督的物體)— 7.6% 像素
  - **baseplate** = d-3dgs 前景 ∖ digger(GT 有、監督裡從來沒有)— 9.5%
  - **background** = 其餘(白)— 82.9%
- 區域 PSNR 只算 mask 內像素(沒有白色像素灌水,所以數值整體比全幀低)。

## 結果

| 模型 | full | **digger** | baseplate | background |
|---|---:|---:|---:|---:|
| Vanilla SC-GS | 11.43 | 10.66 | 5.27 | 13.64 |
| F1 warm-start | 11.55 | 10.87 | 5.42 | 13.70 |
| F2 frozen-canon + deform-MLP | 11.89 | **3.59** | 4.81 | **62.91** |
| **Ours** | 20.64 | **14.90** | 14.70 | 24.80 |

(ours full 20.64 vs 文件 20.35:checkpoint/rotfix 標籤差,baseline 三個數字皆精準重現。)

## 解讀

1. **質疑部分成立:全幀 +8.6/9.2 dB 確實被區域效應放大。**
   83% 像素是背景,vanilla 的爆炸尖刺污染背景(13.6 dB)、底板全失(5.3 dB),
   兩者把它的全幀分數拖得比物體區差距所暗示的更低。

2. **但核心 claim 站得住:digger 區(雙方都被監督、apples-to-apples)我們仍贏。**
   ours 14.90 vs vanilla 10.66 = **+4.2 dB**,vs F2 = **+11.3 dB**。
   贏不只是底板,但 headline 數字應該改用/併報 digger-only。

3. **意外發現:F2 的「11.89,比 vanilla 好 +0.46」其實是假象。**
   F2 在物體區是**四個模型裡最爛的**(3.59 dB —— deform-MLP diverge 後物體基本
   渲染不出來),它全幀贏 vanilla 純粹因為渲出乾淨白背景(62.9 dB)。
   → fairness 結論反而**更強**:同一顆凍結 canonical 給 deform-MLP,物體直接崩;
   給我們的 SE(3)+LBS,物體 14.9 dB。motion model 才是貢獻,證據比之前更乾淨。

4. **底板數字本身也是故事的一部分(要誠實講):**
   我們底板 14.7 dB 來自「body part = 不動」的歸納偏置 —— canonical 裡的靜止
   結構被原封保留。這不是作弊,是 claim 本身(「不該動的不要動」),但 poster
   上要把它跟 digger-only 數字分開報,不要混在全幀 PSNR 裡。

## Poster 上怎麼用

- 重建表加一欄 **digger-only PSNR**:11 → 15 dB(+4.2),並標注全幀數字含
  底板/背景效應。
- F2 那格可以加註「object region 3.6 dB(diverged)」—— 讓 fairness 論證
  從「+8.5 dB」升級成「同 canonical 下 deform-MLP 物體直接崩」。
- 這張表放 appendix,被問「贏在底板?」時直接指過去。
