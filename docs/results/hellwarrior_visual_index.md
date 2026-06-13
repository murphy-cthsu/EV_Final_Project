# Hellwarrior 可視化總索引(2026-06-13)

> 所有 hellwarrior 圖檔一覽,按用途分組。內嵌主圖,完整清單見表。
> 文字分析:`hellwarrior_exp.md`。生成腳本標在每組末。

---

## A. 重建品質(SV4D | clean GT | ours)

3 欄 gallery keyframe(t=10,4 視角):右欄針狀 fuzz = SV4D 噪音「顯形」。

![](../../runs_aux/hellwarrior_gallery/gallery_keyframes.png)

- 逐視角動畫:`runs_aux/hellwarrior_gallery/gallery_v{00,07,14,28}.gif`
- 腳本:`scripts/render_hellwarrior_gallery.py`

## A2. vs vanilla SC-GS(joint baseline,與 lego 對稱)

每列:SV4D | clean GT | vanilla SC-GS(12.7 dB,煙霧塗抹) | ours(13.5 dB)。
gap 只有 +0.83 dB(lego_v2 是 +8.9)→ 監督噪音壓過 method 選擇。

![](../../runs_aux/hellwarrior_vanilla_compare/vanilla_vs_ours_keyframes.png)

vanilla vs 自己的 SV4D 監督(held-out 時刻):身體輪廓在、非全崩,但仍模糊 ~13.5 dB。

![](../../runs_aux/hellwarrior_vanilla_compare/vanilla_fits_sv4d_heldout.png)

- GIF:`runs_aux/hellwarrior_vanilla_compare/vanilla_vs_ours_v0.gif`
- 腳本:`scripts/render_hw_vanilla_compare.py`、`render_hw_vanilla_vs_sv4d.py`
- ⚠️ SC-GS 內部 log 的 "Best PSNR 25.46" 不可比(不同 eval 路徑),以一致管線數字為準

## B. Oracle gap(supervision damage,★poster 主打)

Novel-pose 軌道:左 = floor(乾淨監督 22.7 dB)、右 = ours(SV4D 13.5 dB)。
同 canonical、同動作模組、同 pose,唯一差別是監督乾淨度 = 9.2 dB gap 的視覺形式。

![](../../runs_aux/hellwarrior_gallery/orbit_frame05.png)

- 動畫:`runs_aux/hellwarrior_gallery/orbit_novel.gif`(21 幀)

## C. Reliability-weighted supervision(refine 嘗試,診斷→重建閉環)

control vs β=3 vs β=5(t=10,4 視角):β 越大 fuzz 略收(+0.5 dB),
但仍遠離 lego —— 數據天花板。

![](../../runs_aux/hellwarrior_variant_compare/variant_keyframes.png)

Per-azimuth PSNR:reliability-weighting 整條曲線抬 ~+0.5 dB,cone 形狀保留。

![](../../runs_aux/hellwarrior_variant_compare/per_azimuth_psnr.png)

- 腳本:`scripts/compare_hellwarrior_variants.py`

## D. Mask / Gaussian guidance(高斯有沒有被好好 guide)

6 欄 panel(監督幀 | gt alpha | 渲染 alpha | disagreement | Stage B motion mask | 高斯 part 投影):

![](../../runs_aux/mask_guidance_hellwarrior_hellwarrior_cleancanon_L2_p4/panel_v00_t10.png)

Silhouette-IoU vs azimuth(又一個 cone,純形狀 GT-free):

![](../../runs_aux/mask_guidance_hellwarrior_hellwarrior_cleancanon_L2_p4/iou_vs_azimuth.png)

- panel 全套:`runs_aux/mask_guidance_hellwarrior_*/panel_v{00,07,14,28,42}_t{00,10}.png`
- 腳本:`scripts/viz_mask_guidance.py`

## E. 3D-projection-prompted SAM-2(part 分割原型)

身分在 3D 定一次(左右肢天然分開)→ 投影當 SAM prompt(左=raw 投影,右=SAM refined):

![](../../runs_aux/sam3dprompt_hellwarrior/overlay_v00_t00.png)

- 全套:`runs_aux/sam3dprompt_hellwarrior/overlay_v{00,14,28,42}_t{00,10}.png`
- 腳本:`scripts/sam2_prompt_from_3d.py`

## F. 診斷儀器(跨場景,含 hellwarrior)

GAP-2(R_clean 平坦 → cone 屬於 SV4D 不是參考):

![](../../runs_aux/gap2_reference_floor.png)

SED + FVD/FV4D(hellwarrior 全項比 lego 差):

![](../../runs_aux/instruments_sed_fv4d.png)

---

## 完整檔案清單

| 組 | 路徑 | 內容 |
|---|---|---|
| A | `runs_aux/hellwarrior_gallery/gallery_{keyframes.png,v*.gif}` | 3 欄重建 gallery |
| B | `runs_aux/hellwarrior_gallery/orbit_novel.gif` | oracle vs ours 軌道 |
| C | `runs_aux/hellwarrior_variant_compare/{variant_keyframes,per_azimuth_psnr}.png` | reliability sweep |
| D | `runs_aux/mask_guidance_hellwarrior_*/...` | guidance panel ×10 + IoU cone |
| E | `runs_aux/sam3dprompt_hellwarrior/overlay_*.png` | SAM 3D-prompt ×8 |
| F | `runs_aux/{gap2_reference_floor,instruments_sed_fv4d}.png` | 診斷儀器 |
