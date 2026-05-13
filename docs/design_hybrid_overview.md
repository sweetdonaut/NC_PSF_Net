# NC_PSF_Net 快速概覽（Hybrid 設計）

> **3 分鐘入門指南**。詳細推導與設計理由請循 reference 連結回到完整的 [`design_hybrid.md`](design_hybrid.md)。

---

## 1. 專案做什麼

晶圓檢測，多站點製程。**只在後站找新出現的 PSF defect，忽略所有前站延續來的東西**。

輸入兩張同位置影像（前站、後站），每張 2 通道（target + reference）。模型輸出後站新 anomaly 的 pixel-level heatmap。

> 詳細場景 → [`design_hybrid.md` §3](design_hybrid.md#3-背景與動機)

---

## 2. 核心策略：Hybrid

| | 內容 | 原因 |
|---|---|---|
| 底圖 | **真實**前站影像 + 機台篩選的「乾淨」後站影像 | 拿到真實前站 defect 外觀 + 真實跨站 noise 結構 |
| 監督訊號 | dataloader **動態 inpaint** 合成 PSF defect | 沒有真實人工標註可用；inpaint 位置即 GT |
| 訓練流程 | 強制四錨點 + 9 種 case 機制 | 保證模型學到「只看 prev_T、忽略 prev_R」的判斷規則 |

「為什麼是 Hybrid 不是純 A 或純 B」 → [`design_hybrid.md` §1](design_hybrid.md#1-為什麼是-hybrid)

純 A / 純 B 的歷史紀錄：[`archive/design_strategyA.md`](archive/design_strategyA.md)、[`archive/design_strategyB.md`](archive/design_strategyB.md)

---

## 3. 核心判斷規則（一行）

```
anomaly = (next_T == 1) AND (next_R == 0) AND (prev_T == 0)
```

`prev_R` **不在規則中**。模型訓練的核心就是學會這個非對稱性。

> 推導 → [`design_hybrid.md` §4.3](design_hybrid.md#43-模型要學的精確判斷規則)

---

## 4. 模型輸入規格

```
input: 4 channels = [prev_T, prev_R, next_T, next_R]
output: 2 channels softmax → channel 1 是 anomaly probability
```

模型架構是標準 U-Net + SPPF + SEBlock（沿用 Background_Removal_Net），唯一改動是第一層 conv 接 4 通道。

> 模型細節 → [`design_hybrid.md` §7](design_hybrid.md#7-模型架構)

---

## 5. 9 種訓練 case 與 4 錨點

case 命名規則：`<前站pattern>→<後站pattern>`，pattern 是 2 位元 (T, R)。

**正樣本**（GT=1）：`00→10`、`01→10`  
**負樣本**（GT=0）：`10→10`、`01→01`、`11→11`、`00→01`、`00→11`、`11→10`、`11→01`

**強制四錨點機制**：每個 patch 至少各放一個 `00→10`、`01→10`、`10→10`、`11→10` —— 這四個在後站站內看**長得一模一樣**（都是 T-only），但 prev_T 不同，GT 也不同。模型只能透過 `prev_T` 區分。

> 16 種 case 完整評估 → [`design_hybrid.md` §5](design_hybrid.md#5-case-分析hybrid-下的雙來源)  
> 9 種 case 完整清單 → [`design_hybrid.md` §6](design_hybrid.md#6-最終-9-種訓練-case與策略-b-完全相同)

---

## 6. 資料目錄

```
data/
├── train/{prev,next}/   200 對 (2, H, W) CHW float32 tiff
├── val/{prev,next}/      25 對
├── test/{prev,next}/     25 對
└── manifest.json
```

**沒有 GT mask 檔**。GT 由 dataloader 在每次 `__getitem__` 動態合成（基於本次 inpaint 的位置）。

「這不是 unsupervised；是 synthetic supervised」—— 這個分辨很重要：

> 完整解釋 → [`design_hybrid.md` §4.4–§4.9](design_hybrid.md#44-data-目錄與檔案格式)（六個子節：目錄格式、影像內容、GT 機制、設計理由、能/不能驗證的、合成監督本質）

---

## 7. 程式碼地圖

| 檔案 | 角色 |
|---|---|
| `src_core/model.py` | `SegmentationNetwork(in_channels=4, out_channels=2)` |
| `src_core/dataloader.py` | 9 種 CASES 字典、4 錨點機制、動態 inpaint |
| `src_core/loss.py` | Focal Loss + cosine gamma schedule（沿用 Background_Removal_Net） |
| `src_core/trainer.py` | 訓練流程 + `evaluate_synthetic` |
| `src_core/inference.py` | 4 通道 sliding-window 推論 + 視覺化 |
| `src_core/eval_synthetic.py` | 在 test set 上動態合成 + 計算 AUROC |
| `src_core/generate_psf.py` | PSF defect 生成（沿用 Background_Removal_Net） |
| `src_core/defects/*.yaml` | PSF type 設定（type1~3 純量、type4_vector_strong 向量） |
| `src_core/train.sh` | 預設用 `type4_vector_strong` 訓練 |
| `src_core/inference.sh` | 預設讀 `mvp_vector` checkpoint |
| `scripts/build_paired_dataset.py` | 資料準備（Phase 1: 合成；Phase 2: 真實） |

---

## 8. 怎麼跑

```bash
# 1. 準備資料（Phase 1：合成）
python scripts/build_paired_dataset.py

# 2. 訓練（預設 vector PSF，~10 分鐘 on RTX 5070 Ti）
cd src_core && bash train.sh

# 3. 在 test set 上產出視覺化（每對 ~1 秒）
bash inference.sh

# 4. 在 test set 上跑帶 GT 的 eval（用 eval_synthetic.py，會計算 pixel AUROC）
python eval_synthetic.py \
    --model_path ../checkpoints/mvp_vector/NCPSF_lr0.001_ep20_bs16_128x128.pth \
    --prev_test_path ../data/test/prev --next_test_path ../data/test/next \
    --output_dir ../output/mvp_vector_eval \
    --img_format tiff --defect_mode psf --psf_type type4_vector_strong \
    --psf_pool_size 200 --psf_pool_workers 6 \
    --num_defects_range 6 12 --n_images 25 --gpu_id 0 --seed 12345
```

---

## 9. 目前狀態與 Phase 規劃

| Phase | 內容 | 狀態 |
|---|---|---|
| **Phase 1** | 程式碼 2 通道重構 + 合成資料訓練 | ✅ 完成（commit `bace3b2`），val/test AUROC = 1.0000 |
| **Phase 2** | 改用真實前後站影像當底圖 | ⏳ 等真實資料 + 機台篩選結果到位；只需重寫 `scripts/build_paired_dataset.py` |

Phase 2 的目標**不是**取得真實 anomaly 效能（那需要人工標註資料），而是讓 model 看到真實 noise 分布 + 真實前站 defect 外觀，縮小 domain gap。

> 完整 limitations 與後續方向 → [`design_hybrid.md` §12](design_hybrid.md#12-已知限制與後續方向)

---

## 10. 進階閱讀路線

依優先順序：

1. **核心邏輯** → [`design_hybrid.md` §1–§4](design_hybrid.md#1-為什麼是-hybrid)（為什麼 hybrid、判斷規則、資料設計）
2. **訓練機制** → [`design_hybrid.md` §5–§6](design_hybrid.md#5-case-分析hybrid-下的雙來源)（case 分析、四錨點）
3. **實作細節** → [`design_hybrid.md` §13](design_hybrid.md#13-實作-checklist從目前-code-演化到-hybrid)（檔案層級的改動清單）
4. **未來工作** → [`design_hybrid.md` §9](design_hybrid.md#9-乾淨後站篩選機制) + [`§10`](design_hybrid.md#10-inpaint-避撞策略)（Phase 2 的篩選與避撞）
5. **設計演化** → [`archive/`](archive/)（v1 3 通道、純 A、純 B 各自的設計筆記）
