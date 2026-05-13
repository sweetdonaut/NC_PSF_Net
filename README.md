# NC_PSF_Net

晶圓檢測多站點製程下的 PSF defect 偵測。輸入前站、後站同位置影像對（每張 2 通道：target + reference，共 4 通道），輸出後站**新增** PSF defect 的 pixel-level heatmap。前站延續來的 defect 要忽略。

## 入口文件

- **完整設計（推薦）**：[`docs/design_hybrid_refresh.md`](docs/design_hybrid_refresh.md)
- 3 分鐘快速入門：[`docs/design_hybrid_overview.md`](docs/design_hybrid_overview.md)

> 舊版 [`docs/design_hybrid.md`](docs/design_hybrid.md) 暫時保留供對照；overview 內的章節連結仍指向舊版，refresh promote 後會一次性對齊。

## 怎麼跑

```bash
# 1. 準備資料（Phase 1：從 BRN grayscale 合成乾淨底圖）
python scripts/build_paired_dataset.py

# 2. 訓練（預設 vector PSF，~10 分鐘 on RTX 5070 Ti）
cd src_core && bash train.sh

# 3a. 在 test set 上產出視覺化（每對 ~1 秒）
bash inference.sh

# 3b. 在 test set 上跑帶 GT 的 eval（會計算 pixel AUROC）
python eval_synthetic.py \
    --model_path ../checkpoints/mvp_vector/NCPSF_lr0.001_ep20_bs16_128x128.pth \
    --prev_test_path ../data/test/prev --next_test_path ../data/test/next \
    --output_dir ../output/mvp_vector_eval \
    --img_format tiff --defect_mode psf --psf_type type4_vector_strong \
    --psf_pool_size 200 --psf_pool_workers 6 \
    --n_images 25 --gpu_id 0 --seed 12345
```

## 目前狀態

| Phase | 內容 | 狀態 |
|---|---|---|
| Phase 1 | 程式碼 2 通道重構 + 合成資料訓練 | 完成（val/test AUROC = 1.0000）|
| Phase 2 | 改用真實前後站影像當底圖 | 等真實資料 + 機台篩選結果到位 |
