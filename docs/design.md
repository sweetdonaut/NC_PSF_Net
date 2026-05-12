# NC_PSF_Net 設計文件

## 1. 背景與動機

### 1.1 場景描述

晶片製造採多站點流程。同一片晶圓會依序通過多個處理站點，每個站點各自可能產生瑕疵 (defects)：

- **前站 (previous station)**：先一步處理，可能已產生 defects
- **後站 (next station)**：在前站之後處理，可能產生新的 defects

實務上，後站的影像會**同時包含**前站既有的 defects（物理上延續存在）與後站新增的 defects。我們的檢測目標**只關心「後站新增」的部分**，但前站的 defects 對 rule-based 演算法構成大量干擾。

### 1.2 為什麼困難

前後站經過不同製程，影像在 noise、紋理、對比度上差距非常大。傳統 rule-based 方法依賴影像統計特性，前後站的差異會直接擊穿這類方法的假設。

### 1.3 核心策略

放棄讓模型「理解」複雜的 noise，改為：**讓 U-Net 同時看前後站影像，學會比對兩者，只標出後站獨有的 PSF defect**。

把所有困難的 noise 處理交給 U-Net 自動學習，模型只需要專注於「找出後站獨有的新 PSF defect」。

---

## 2. 資料結構

### 2.1 die-to-die 影像規格

每張前/後站影像本身是 **3 通道灰階**（die-to-die comparison）：

- `target`：目標 die
- `ref1`：相鄰參考 die #1
- `ref2`：相鄰參考 die #2

前後站對應同一片晶圓的同一位置，**前後站之間像素對齊**（MVP 假設）。

模型輸入 = 前站 3 通道 + 後站 3 通道 concat = **6 通道**：

```
[prev_T, prev_R1, prev_R2, next_T, next_R1, next_R2]
```

### 2.2 die-to-die 黃金法則

對單站三通道：**只有 T-only (100) pattern 才是 anomaly 候選**。

若 T 上有 defect 但至少一個 ref 也有，即視為非 anomaly（可能是 wafer 級污染、相鄰 die 同位置巧合，或系統性製程現象）。

這條規則沿襲自 `Background_Removal_Net` 的設計，在 NC_PSF_Net 中**獨立應用於前站與後站**，並再加上「跨站對比」這一層判斷。

### 2.3 模型最終要學的兩層邏輯

1. **站內 die-to-die 黃金法則**：T-only 才是 anomaly 候選
2. **跨站對比**：anomaly 候選還必須是「前站沒有」，才是真正的後站新 anomaly

---

## 3. 訓練資料合成邏輯

### 3.1 策略選擇（策略 A）

採用**策略 A**：用真實前後站影像對 + 合成 PSF defect inpaint。

- 前站影像：真實影像（可帶有真實的前站既有 defects）
- 後站影像：真實「無新 defect」的對應影像（自然包含前站延續過來的 defects）
- 在前後站上 inpaint 合成 PSF defects，作為訓練 signal

備案策略 B 見 §7。

### 3.2 14 種 defect case 完整列表

每個合成 defect 屬於 14 種 case 之一，由「來源」（前站既有 / 後站新增）與三通道 pattern 決定：

| Case | 來源 | 前站 (T R1 R2) | 後站 (T R1 R2) | GT |
|------|------|----------------|----------------|----|
| A1 | 前站既有 | 1 0 0 | 1 0 0 | 0 |
| A2 | 前站既有 | 0 1 0 | 0 1 0 | 0 |
| A3 | 前站既有 | 0 0 1 | 0 0 1 | 0 |
| A4 | 前站既有 | 1 1 0 | 1 1 0 | 0 |
| A5 | 前站既有 | 1 0 1 | 1 0 1 | 0 |
| A6 | 前站既有 | 0 1 1 | 0 1 1 | 0 |
| A7 | 前站既有 | 1 1 1 | 1 1 1 | 0 |
| **B1** | **後站新增** | **0 0 0** | **1 0 0** | **1** ⭐ |
| B2 | 後站新增 | 0 0 0 | 0 1 0 | 0 |
| B3 | 後站新增 | 0 0 0 | 0 0 1 | 0 |
| B4 | 後站新增 | 0 0 0 | 1 1 0 | 0 |
| B5 | 後站新增 | 0 0 0 | 1 0 1 | 0 |
| B6 | 後站新增 | 0 0 0 | 0 1 1 | 0 |
| B7 | 後站新增 | 0 0 0 | 1 1 1 | 0 |

**只有 B1 是 GT=1**。其他 13 種都是要學會忽略的負樣本。

### 3.3 Case 角色分析

#### 絕對核心（每個 patch 強制各放一個）

- **B1**：唯一正樣本
- **A1**：唯一強迫模型「必須跨站對比」的 case
  - 後站站內看 A1 是 T-only（最像 anomaly）
  - 但前站也有同樣 pattern → 必須跨站才能判斷不是新 defect
  - 沒有 A1，模型會 shortcut 退化成「找後站站內 T-only」，跨站對比形同虛設

#### 次要但需要（防止 shortcut）

- **B4 / B5 / B7**：「T 有但 ref 也有」的後站新增 case
  - 防止模型學成「跨站獨有就標 1」這個 shortcut
  - inference 時若真的出現後站新增的多通道 defect，必須能正確判斷不是 anomaly

#### 填補分布

- A2~A7、B2、B3、B6：覆蓋 inference 時可能遇到的所有 case 分布

### 3.4 合成流程

每個 patch：

1. **50% 機率完全不放 defect** → GT 全 0
   讓模型學到「正常情況輸出全黑」

2. 否則：
   - 隨機 N 個 defects（範圍預設 3~8，可調）
   - **強制其中 1 個是 A1**
   - **強制其中 1 個是 B1**
   - 剩下 N-2 個從 14 種 case 均勻隨機抽

### 3.5 與 Background_Removal_Net 的對應

原專案的單站智慧貼邏輯 (`target_only / only_ref1 / only_ref2 / all`) 在 NC_PSF_Net 中**獨立應用於前後兩站**：

- 前站既有 defects 用智慧貼分配 T/R1/R2 子集 → 對應通道 mirror 到後站（物理上延續）
- 後站新增 defects 用智慧貼分配 T/R1/R2 子集 → 只有 T-only (B1) 標為 GT

7 種前站既有 × 7 種後站新增中的 1 種正樣本 + 6 種負樣本 = 14 種 case。

### 3.6 不一致 case（暫不處理）

物理上罕見但可能的 case：

- 前站 100、後站 110（前站既有 + 後站該位置又疊加新增）
- 前站 100、後站 100 但強度不同（製程造成 defect 演化）

這些是「同位置兩個 defect 疊加」或「同 defect 變化」。MVP **暫不考慮**，理由：

1. 隨機位置抽取下重疊機率極低
2. 14 種 case 已涵蓋主要訓練訊號
3. 加進來會讓 dataloader 複雜度暴增

留待後續若需要 robustness 訓練再加入。

---

## 4. PSF Defect 生成

直接共用 `Background_Removal_Net/src_core/generate_psf.py`：

- 環形光瞳 (annular aperture) + Zernike 像差 → FFT → Poisson/Gaussian noise → connected-peak 清理
- 支援純量 PSF 與 Richards-Wolf 向量 PSF
- 預先用 multiprocess pool 產生 PSF defect pool（`SeedSequence.spawn()` 確保非重疊隨機流）
- 多種 PSF type（type1~4、low_intensity）可同時混用

YAML config 完全相同，可直接複用 `defects/*.yaml`。

---

## 5. 模型架構

### 5.1 MVP：沿用原 U-Net

```python
SegmentationNetwork(in_channels=6, out_channels=2)
```

- 與 `Background_Removal_Net` 相同的 encoder-decoder + SPPF + SEBlock
- 唯一改動：第一層 conv 接受 6 通道輸入

### 5.2 為什麼第一版不用 Siamese / cross-attention

- 300 張資料 + 高度合成訓練的情境下，架構創新的邊際效益遠低於確保資料 pipeline 正確
- 6 通道 concat 讓 encoder 自己學前後站對比，U-Net 容量足夠
- 先驗證 baseline，後續再升級

後續可能升級方向見 §8。

---

## 6. 訓練策略

### 6.1 Loss

Focal Loss with cosine gamma schedule（沿用原專案）：

- `alpha = 0.75`
- `gamma_start → gamma_end`，cosine 衰減

### 6.2 資料增強

- patch 切割本身就是空間增強
- 每次 `__getitem__` 都重新隨機合成 defects → 等效無限增強
- MVP 不額外加 flip / rotate，先驗證 baseline

### 6.3 資料切分（300 張）

```
240 train / 30 val / 30 test  (8:1:1)
```

val / test 的 GT 也是合成（相同的 14 種 case），主要用於監控訓練收斂。

**真實效能驗證需另外收集人工標註資料**（後續工作）。

### 6.4 期望規模

- 影像大小假設 ~1024×1024，patch 128×128 → 每張可切 ~60 patches
- 300 張 × 60 patches × 隨機合成 → 每 epoch 等效樣本數 ≫ 20k
- 與原專案 MVTec 訓練規模相當，預期可收斂

---

## 7. 策略 B：前後站皆合成（備案）

目前採用策略 A（用真實前站影像）。備案策略 B：

- 前後站都用乾淨基底影像
- 14 種 case 完全用合成 PSF 生成（包括 A 類延續關係）

**優點**：

- 不依賴「乾淨後站影像」的真實供應
- 完全可控的 ground truth
- 不依賴前後站精準配對

**缺點**：

- 缺乏真實前站 defect 的外觀變化
- 泛化能力可能受限

若策略 A 因資料量或 GT 純淨度問題受挫，可切換至策略 B。

---

## 8. 後續可能擴充

### 8.1 架構升級

- **Siamese twin encoder**：前後站各一 encoder（共享權重），decoder 端融合
- **Cross-attention**：在 feature level 對齊前後站
- **Deformable / correlation**：處理輕微 misalignment

### 8.2 資料設計擴充

- 前後站不一致 pattern (前 100 後 110 等疊加 case)
- 同 defect 在前後站強度演化
- partial leak（後站新增 defect 弱漏到後站 ref）
- 強制 flip / rotate / 強度抖動等傳統增強

### 8.3 真實標註資料

- 收集人工標註的後站新 defect mask
- 用於真實效能驗證
- 後期可加入 supervised fine-tuning

---

## 9. 已知限制

1. **像素對齊假設**：MVP 假設前後站像素級對齊，實際資料可能需要預配準
2. **Domain gap**：合成 PSF vs 真實 defect 的外觀差異
3. **資料量**：300 張底圖偏少，依賴合成的「無限增強」
4. **無 partial leak**：die-to-die 製程不完美未模擬
5. **真實效能未驗證**：合成 GT 上的 AUROC 不等於真實場景效能

---

## 10. 與 Background_Removal_Net 的關係

NC_PSF_Net **直接共用** Background_Removal_Net 的：

- `generate_psf.py`（PSF 生成）
- `gaussian.py`（局部 inpaint 工具）
- `loss.py`（Focal Loss）
- `defects/*.yaml`（PSF configs）

**重新撰寫**的部分：

- `dataloader.py`：前後站配對 + 14 種 case 合成
- `model.py`：`in_channels=6`
- `trainer.py`：輸入鍵名與通道調整
- `inference.py`：前後站雙影像輸入

核心邏輯上，NC_PSF_Net 等於「`Background_Removal_Net` 的智慧貼邏輯 × 2 站 + 跨站對比」。
