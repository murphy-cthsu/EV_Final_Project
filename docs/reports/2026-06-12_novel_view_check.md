# Novel-view 幾何正確性對比 + SC-GS 實作 sanity check

> 2026-06-12。承接 region-PSNR 分析(`2026-06-12_region_psnr.md`):
> 在訓練視角上 digger-only 只贏 +4.2 dB,那 novel view 呢?
> 另外驗證 vanilla SC-GS 的爆炸不是我們實作/設定的問題。
>
> 腳本:`scripts/render_novel_view_compare.py` → `runs_aux/novel_view_compare/`。

## 設定

- 兩個模型都只用 lego_v2 的 5 視角(elev0 az 0/60/120/180/240)訓練。
- Novel view = lego_v3 grid 裡不在訓練集的 pose(world frame 已驗證一致,
  camera center 完全重合),GT = `lego_v3_d3dgs_ref`(乾淨 d-3dgs)。
- 6 個 novel view:az 150/210/270/330(elev0,插值方位)+ elev20_az180、
  elev30_az120(高度外插)。

## 結果:novel view 我們大幅贏,且是幾何性的

| novel view | vanilla SC-GS | ours | Δ |
|---|---:|---:|---:|
| elev0 az150 | 11.55 | 19.46 | +7.9 |
| elev0 az210 | 11.70 | 21.29 | +9.6 |
| elev0 az270 | 10.74 | 19.02 | +8.3 |
| elev0 az330 | 11.10 | 18.57 | +7.5 |
| elev20 az180 | 11.03 | 20.30 | +9.3 |
| elev30 az120 | 11.17 | 19.82 | +8.7 |
| **mean** | **11.22** | **19.74** | **+8.5** |

Keyframe grid(t=10;每列 = GT | vanilla | ours):

![](../../runs_aux/novel_view_compare/novel_keyframe_grid.png)

逐視角動畫:`runs_aux/novel_view_compare/novel_elev_0_az_150.gif` 等 6 支。

**質性觀察(肉眼):**
- vanilla 在每個 novel view 都是災難:黑色爆炸碎片、物體碎裂,az210/az270
  幾乎只剩殘片 —— 不是「品質差」而是**幾何不存在**。
- ours 在所有 novel view 物體位置、姿態、底板都正確,差距只在銳利度
  (比 GT 糊)。**幾何正確性整條贏,差的只是外觀細節。**
- 注意 novel-view PSNR(~19.7)幾乎不低於訓練視角 PSNR(20.6)——
  凍結 canonical 讓我們的表現本質上 view-uniform;vanilla 則在訓練視角
  本身就已經崩(11.4),novel view 一樣崩(11.2)。

## 跟 region-PSNR 表合起來的完整圖像

- 訓練視角 digger-only:+4.2 dB(贏,但溫和)。
- Novel view 全幀:+8.5 dB,且質性上是「有幾何 vs 沒幾何」。
- → poster 的正確 claim:**「在 noisy VGM 監督下,凍結+結構化動作得到
  view-consistent 的正確幾何;joint baseline 連訓練視角都過擬合成碎片」**。
  不 claim 銳利度/外觀品質(那是 CAT4D 的仗,我們不打)。

## Sanity check:SC-GS 實作本身沒問題

用完全相同的 SC-GS 程式 + 訓練旗標,訓在**乾淨 D-NeRF lego**(標準資料):

```
python third_party/SC-GS/train_gui.py --source_path data/dnerf/lego \
  --model_path outputs/custom/dnerf_lego_vanilla_sanity \
  --deform_type node --node_num 512 --hyper_dim 8 \
  --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame \
  --resolution 1 --W 800 --H 800 --iterations 20000
```

- held-out test:**best PSNR 25.18 / SSIM 0.950 / LPIPS 0.053**(20k 跑完)。
- 肉眼確認(5 個 held-out test view × 不同時刻,GT | render):
  幾何完整、無爆炸、無黑色尖刺,逐幀 21–24 dB —— 正常 SC-GS 水準。

![](../../runs_aux/novel_view_compare/dnerf_lego_sanity_grid.png)

- 模型:`outputs/custom/dnerf_lego_vanilla_sanity_node/`(20k iters,~22 min)。
- **結論:vanilla 在 lego_v2 上的爆炸來自 SV4D noisy 監督(+ 只有 5 視角),
  不是我們的 SC-GS 設定或實作問題。** baseline 沒有被我們弄壞,
  novel-view 對比(上表 +8.5 dB)可以放心引用。
