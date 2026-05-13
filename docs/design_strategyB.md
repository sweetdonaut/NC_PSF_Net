# NC_PSF_Net 設計文件 v2（2 通道版，策略 B）

> **本文件 status**：第二版設計。生產環境確認每站影像只有 target + 1 reference（共 2 通道），原 v1 假設的 3 通道規格已作廢。
>
> **訓練策略**：策略 B（fully synthetic baseline）—— 底圖完全是合成乾淨影像，所有 defect 訊號透過 inpaint 提供。策略 A 的對應分析另開文件。
>
> **與 `design.md`（v1）的關係**：v1 為 3 通道版本歷史紀錄保留；本文件取代 v1 作為當前實作依據。
>
> **命名規則**：本文件用 `前pattern→後pattern` 當 case 代號（例如 `00→10`），不再使用 ABCD 字母分類。pattern 為 2 位元 `(T, R)`，例如 `10` 表示 T die 有 defect、R die 沒有。

---

## 1. 規格變化摘要

| 項目 | v1（已作廢） | v2（本文件） |
|---|---|---|
| 每站通道 | 3 (T, R1, R2) | **2 (T, R)** |
| 模型輸入 | 6 channels | **4 channels** `[prev_T, prev_R, next_T, next_R]` |
| 單站非空 pattern | 7 種 | **3 種** (`10`/`01`/`11`) |
| 前後站完整組合空間 | 8 × 8 = 64 | **4 × 4 = 16** |
| 訓練合成 case 數 | 14 種 | **9 種** |
| 強制機制 | 每 patch ≥ 1 個錨點 | **每 patch 至少各 1 個「後站 T-only 家族」四錨點** |

其他設計（PSF 生成器、loss、訓練流程、模型架構骨幹）**完全沿用 v1**。

---

## 2. 背景與動機

晶片製造採多站點流程。同一片晶圓會依序通過多個處理站點，每個站點都可能產生 defects：

- **前站**：先一步處理，可能已有 defects
- **後站**：在前站之後處理，可能產生新的 defects

後站影像會同時包含前站既有 defects（物理上延續存在）與後站新增 defects。我們**只關心後站新增**的部分，但前站延續 defects 對 rule-based 算法構成大量干擾。

核心策略不變：**讓 U-Net 同時看前後站影像，學會對比兩者，只標出後站獨有的 PSF defect**。

---

## 3. 資料結構

### 3.1 die-to-die 影像規格

每張前/後站影像是 **2 通道灰階**（die-to-die comparison）：

- `target` (T)：目標 die
- `ref` (R)：相鄰參考 die

前後站對應同一片晶圓的同一位置，**前後站之間像素對齊**（MVP 假設）。

模型輸入 = 前後站 concat = **4 通道**：

```
[prev_T, prev_R, next_T, next_R]
```

### 3.2 die-to-die 黃金法則

對單站二通道：**只有 T-only (`10`) pattern 才是 anomaly 候選**。

- `10` (T-only) → anomaly 候選
- `01` (R-only) → ref die 缺陷，**不算**
- `11` (T+R) → 兩 die 同位置都有，可能是 wafer 級或巧合，**不算**

這條規則沿襲自 Background_Removal_Net 設計，在 NC_PSF_Net 中**獨立應用於前站與後站**，並再加上「跨站對比」這一層判斷。

### 3.3 模型要學的精確判斷規則

合併「站內 die-to-die 黃金法則」與「跨站對比」後，「該標的 anomaly」可以寫成一個精確判斷式：

> **anomaly = (`next_T = 1`) AND (`next_R = 0`) AND (`prev_T = 0`)**
>
> 即：後站站內 T-only **且** T 通道在前站乾淨

**重點：`prev_R` 不在判斷規則中**。R 通道在前站的狀態與「後站 T 上是否為新 anomaly」邏輯上無關 —— 這個對稱性破壞是後面 §4 為什麼 `01→10` 必須訓練、`10→01` 不必訓練的根本原因。

模型實際上要學會的不是「找出後站獨有的東西」，而是「**判斷後站站內 T-only 是否在前站 T 通道也存在**」。R 通道（前後站皆然）提供 die-to-die 比對的視覺特徵，但不直接決定標不標。

---

## 4. Case 分析：從 16 種完整組合導出 9 種訓練 case

### 4.1 判斷框架

訓練影像 = **底圖 + 主動 inpaint 的 defects**。任何 case 的 training signal 有兩種來源：

- **來源 A（底圖隱含）**：底圖本身的物理特性自然產生這個 case → 模型透過大量背景像素自動學到
- **來源 B（必須 inpaint）**：底圖永遠不會出現這個 case → 必須合成才能訓練

對每個 case 問三個問題：
1. **inference 會遇到嗎？** 不會 → 不必訓練
2. **底圖會自然產生嗎？** 會 → 不必特別 inpaint
3. **GT 是什麼？** 決定模型輸出應為 0 還是 1

判斷規則：

> **必須 inpaint 的 case = (inference 會遇到) AND (底圖不會自然產生)**

### 4.2 在策略 B 下「底圖」的定義

`build_paired_dataset.py` 產生的底圖 = BRN grayscale → process shift → 乾淨前後站影像對。

**底圖本身完全沒有任何 defects**，只有：
- 跨站 systematic brightness/contrast/noise shift（模擬製程差異）
- 通道間微弱 die noise（模擬不同 die）

→ **策略 B 的底圖只會自然產生 `00→00`（無事件背景）**，其他 15 種 case 都必須透過 inpaint 訓練。

> **重要依賴**：這個結論只在策略 B 下成立。策略 A（用真實前站影像當底圖）下，前站延續類會自然存在於底圖，邏輯會改變，另文件分析。

### 4.3 16 種 case 完整評估表

| Pattern | 製程情況 | inference 遇到？ | 底圖自然產生？ | GT | 結論 |
|---|---|---|---|---|---|
| `00→00` | 正常背景區（無 defect） | 是（大部分像素） | ✅ 是 | 0 | **不必 inpaint** |
| `10→10` | 前站 T-only 缺陷往後傳遞 | 是 | ❌ 否 | 0 | **必須 inpaint** |
| `01→01` | 前站 R-only 缺陷往後傳遞 | 是 | ❌ 否 | 0 | **必須 inpaint** |
| `11→11` | 前站 T+R 缺陷往後傳遞 | 是 | ❌ 否 | 0 | **必須 inpaint** |
| `00→10` | **後站新增 T-only 缺陷（要找的 anomaly）** | 是（**目標！**） | ❌ 否 | **1** | **必須 inpaint** ⭐ |
| `00→01` | 後站新增 R-only 缺陷（ref die） | 是 | ❌ 否 | 0 | **必須 inpaint** |
| `00→11` | 後站新增 T+R 缺陷（wafer 級新事件） | 是 | ❌ 否 | 0 | **必須 inpaint** |
| `10→00` | 前站 T-only 被後站製程移除 | 罕見 | ❌ 否 | 0 | 不必 inpaint（見 §4.4） |
| `01→00` | 前站 R-only 被後站製程移除 | 罕見 | ❌ 否 | 0 | 不必 inpaint |
| `11→00` | 前站 T+R 被後站製程整體移除 | 罕見 | ❌ 否 | 0 | 不必 inpaint |
| `10→01` | 前站 T 缺陷消失 + 後站 R 新增（雙獨立事件） | 罕見 | ❌ 否 | 0 | 不必 inpaint（與 `00→01` 等價，見 §4.4） |
| `01→10` | **前站 R 缺陷消失 + 後站 T 新增 anomaly（雙獨立事件）** | 罕見 | ❌ 否 | **1** | **必須 inpaint** ⭐（見 §4.5） |
| `10→11` | 前站 T-only + 後站該位置 R 又新增缺陷 | 極罕見 | ❌ 否 | 0 | 不必 inpaint（見 §4.4） |
| `01→11` | 前站 R-only + 後站該位置 T 又新增缺陷 | 極罕見 | ❌ 否 | 0 | 不必 inpaint |
| `11→10` | **前站 T+R，後站 R 缺陷被製程選擇性移除** | **是**（製程選擇性移除） | ❌ 否 | 0 | **必須 inpaint** ⚠️ |
| `11→01` | 前站 T+R，後站 T 缺陷被製程選擇性移除 | 是 | ❌ 否 | 0 | **必須 inpaint** |

### 4.4 為什麼這幾類不必 inpaint

#### 消失類（`10→00`, `01→00`, `11→00`）
- inference 遇到時：模型看到「後站該位置視覺上完全乾淨」
- **後站乾淨 = `00→00` 的局部範圍**，模型對乾淨像素的輸出本來就是 0
- 判斷依據是後站影像本身，即使前站有 defect 也不會誘導誤判
- **後站範圍等同 `00→00`，已被底圖自然覆蓋**

#### 「Pattern 不一致」類（`10→01`, `01→10`）— 兩個獨立事件同位置疊加
這兩個 case 不是「同一個 defect 跨 die 跳」，而是兩個獨立物理事件疊加：

- **`10→01`**：前站 T-only defect 在後站消失 + 後站 R die 上新增 defect
- **`01→10`**：前站 R-only defect 在後站消失 + 後站 T die 上新增 defect

兩個獨立事件同位置疊加機率不高，但物理上**確實可能**。判斷規則 §3.3 自然處理它們：

- **`10→01`**：`next_T = 0` → 不滿足規則 → GT=0，與 `00→01` 視覺及邏輯等價，模型透過 `00→01` 訓練自然推廣
- **`01→10`**：`next_T = 1` AND `next_R = 0` AND `prev_T = 0` → **滿足規則 → GT=1**，**必須單獨訓練**（理由見 §4.5）

#### 疊加類（`10→11`, `01→11`）— 前站有 + 後站該位置又新增
- 物理機率 = (per-pixel defect density)²，極低
- inference 可能 100k 個 patch 才碰到一次
- 訓練浪費 capacity，且加入會干擾其他 case 的比例平衡

### 4.5 為什麼 `11→10`、`11→01`、`01→10` 必須加

#### `11→10`（R 缺陷被製程選擇性移除）— 核心
- 物理意義：前站 T+R 都有 defect，後站某個 die（R）上的 defect 被製程清掉
- inference 確實會遇到
- **關鍵**：`11→10` 在後站站內看是 T-only —— **跟 `00→10` 視覺特徵幾乎一樣**
- 模型沒看過 `11→10`，遇到時會被 `00→10` 的「站內 T-only → 標 1」邏輯誤導誤判
- `11→10` 跟 `10→10` 角色對稱：兩者都是「強迫跨站對比」的核心負樣本

#### `11→01`（T 缺陷被製程選擇性移除）— 補完整性
- 物理意義：T+R 後站只剩 R-only
- 後站站內看是 R-only，按 die-to-die 法則本來就不算 anomaly
- 加入主要是維持 case 分布完整，避免模型在「前站 T+R + 後站某個 die 變化」的情境下表現不穩

#### `01→10`（前站 R 消失 + 後站 T 新增）— 正樣本變體 ⭐
這是修正版設計新加入的 case，補上原先的盲區。物理意義：兩個獨立事件同位置疊加 —— 前站 R die 上原本有 defect（已消失）+ 後站 T die 上新增 defect。

**為什麼必須單獨訓練**：四個 case 在後站站內看**視覺完全一樣**（都是 T-only），但 GT 不同：

| Pattern | prev_T | prev_R | GT |
|---|---|---|---|
| `00→10` | 0 | 0 | **1** |
| **`01→10`** | **0** | **1** | **1** |
| `10→10` | 1 | 0 | 0 |
| `11→10` | 1 | 1 | 0 |

模型只能透過 `prev_T` 區分這四個 case。若只訓練 `00→10` + `10→10` + `11→10`（缺 `01→10`），模型很可能學成 shortcut：「**前站任何通道有東西 → 標 0**」。

這個 shortcut 對 `00→10`/`10→10`/`11→10` 三者都對，但對 **`01→10` 會誤判** —— 前站 R 有 defect 但 T 沒有，按 shortcut 規則 → 標 0，但正確答案是 1。

加入 `01→10` 訓練 + 對齊的強制錨點機制，可以逼模型學到正確規則：**只看 `prev_T`，忽略 `prev_R`**。

---

## 5. 最終 9 種訓練 case

| Pattern | 製程情況 | GT | 訓練角色 |
|---|---|---|---|
| `10→10` | 前站 T-only 缺陷往後傳遞 | 0 | 後站 T-only 家族錨點（前 `10`） |
| `01→01` | 前站 R-only 缺陷往後傳遞 | 0 | R-only 延續分布覆蓋 |
| `11→11` | 前站 T+R 缺陷往後傳遞 | 0 | T+R 延續分布覆蓋 |
| `00→10` | 後站新增 T-only 缺陷 | **1** ⭐ | 後站 T-only 家族錨點（前 `00`，正樣本 #1） |
| `01→10` | 前站 R 缺陷消失 + 後站 T 新增（雙獨立事件） | **1** ⭐ | 後站 T-only 家族錨點（前 `01`，正樣本 #2） |
| `00→01` | 後站新增 R-only 缺陷 | 0 | 後站新 R-only |
| `00→11` | 後站新增 T+R 缺陷 | 0 | 防站內 T 誘惑 |
| `11→10` | 前站 T+R，後站 R 缺陷被製程移除 | 0 | 後站 T-only 家族錨點（前 `11`） |
| `11→01` | 前站 T+R，後站 T 缺陷被製程移除 | 0 | T+R 部分移除分布覆蓋 |

### 5.1 強制機制：「後站 T-only 家族」四錨點

**每個 patch 至少各放 1 個 `00→10`、`01→10`、`10→10`、`11→10`**。

這四個 case 在「後站站內看」全部都是 T-only，差別只在前站 pattern：

| Pattern | prev (T R) | GT | 意義 |
|---|---|---|---|
| `00→10` | 0 0 | **1** | 真新增（前站全乾淨） |
| `01→10` | 0 1 | **1** | 真新增（前站只有 R 干擾） |
| `10→10` | 1 0 | 0 | 假新增（前站 T 就有） |
| `11→10` | 1 1 | 0 | 假新增（前站 T+R 都有） |

**唯一能正確區分的訊號是 `prev_T`**。四錨點齊全才能逼模型學到「只看 `prev_T`，忽略 `prev_R`」的正確規則，否則模型會 shortcut 成「前站任何通道有東西就標 0」，在 `01→10` 上系統性誤判。

### 5.2 合成流程

每個 patch：

1. **50% 機率完全不放 defect** → GT 全 0
2. 否則：
   - 隨機 N 個 defects（範圍預設 4~10，因為錨點數從 3 升到 4，下限提高）
   - **強制塞 1 個 `00→10` + 1 個 `01→10` + 1 個 `10→10` + 1 個 `11→10`**（若 N < 4 退化）
   - 剩下 N-4 個從 9 種 case 均勻隨機抽

---

## 6. 模型架構

### 6.1 MVP：沿用 v1 U-Net 骨幹

```python
SegmentationNetwork(in_channels=4, out_channels=2)
```

- 與 Background_Removal_Net 相同的 encoder-decoder + SPPF + SEBlock
- 唯一改動：第一層 conv 接受 4 通道輸入（v1 是 6 通道）

### 6.2 為什麼第一版不用 Siamese / cross-attention

- 300 張資料 + 高度合成訓練的情境下，架構創新邊際效益低
- 4 通道 concat 讓 encoder 自己學跨站對比
- 先驗證 baseline 收斂正確，後續再考慮升級

---

## 7. 訓練策略

### 7.1 Loss

Focal Loss with cosine gamma schedule（沿用 v1）：
- `alpha = 0.75`
- `gamma_start → gamma_end`，cosine 衰減

### 7.2 資料增強

- patch 切割本身就是空間增強
- 每次 `__getitem__` 都重新隨機合成 → 等效無限增強
- MVP 不額外加 flip / rotate

### 7.3 資料切分（300 張）

```
240 train / 30 val / 30 test  (8:1:1)
```

val/test 的 GT 也是合成，主要用於監控訓練收斂。**真實效能驗證需另外收集人工標註資料**。

---

## 8. 策略 B 的明確界定

### 8.1 策略 B 的特性

- 底圖：**完全合成乾淨**前後站影像對（從 BRN grayscale 加 process shift 生成）
- 所有 defect signal：**完全來自合成 inpaint**
- 優點：
  - 不依賴「乾淨後站影像」的真實供應
  - GT 完全可控、不會混入未標記的真實 defect
  - 不依賴精準前後站配對的真實資料
- 缺點：
  - 缺乏真實前站 defect 的外觀變化，泛化能力可能受限
  - 所有 case 都得 inpaint（無法利用真實前站 defect 自然提供持續類訓練）

### 8.2 與策略 A 的差別

**策略 A**：用真實前站影像 + 真實「乾淨」後站影像 + 合成 anomaly 類 defects。

關鍵差異：
- 策略 A 底圖會自然包含持續類（`10→10`、`01→01`、`11→11`）—— 真實前站 defects 延續到後站
- → 持續類可以**不再 inpaint**，省下訓練 capacity 給後站新增與部分移除類
- 但策略 A 要求資料能精準配對「同位置乾淨後站」

策略 A 的第一原則分析另開文件討論。

---

## 9. PSF Defect 生成

直接共用 `Background_Removal_Net/src_core/generate_psf.py`：

- 環形光瞳 + Zernike 像差 → FFT → Poisson/Gaussian noise → connected-peak 清理
- 支援純量 PSF 與 Richards-Wolf 向量 PSF
- 預先用 multiprocess pool 產生 PSF defect pool

YAML config 完全相同，可直接複用 `defects/*.yaml`。

**重要 caveat**：vector PSF 的 `type4_vector.yaml` 原 `intensity_abs = [[8, 12]]` 是設計給「真實光學 PSF amplitude」場景；在 NC_PSF_Net 合成資料上 intensity 是「pixel-perturbation magnitude」，單位不同。建議使用 `type4_vector_strong.yaml`（intensity 60-80）。

---

## 10. 已知限制與後續方向

### 10.1 已知限制
1. **像素對齊假設**：MVP 假設前後站像素級對齊，實際可能需要預配準
2. **Domain gap**：合成 PSF vs 真實 defect 的外觀差異
3. **資料量**：300 張底圖偏少，依賴合成的「無限增強」
4. **無 partial leak**：die-to-die 製程不完美未模擬
5. **真實效能未驗證**：合成 GT 上的 AUROC 不等於真實場景效能

### 10.2 後續方向

#### 10.2.1 切換到策略 A
當有真實前站影像可用時，持續類訓練 capacity 可省下。詳見策略 A 文件。

#### 10.2.2 加入未訓練 case
若實機驗證發現某些 case 表現不穩，可考慮加入：
- 消失類（`10→00`/`01→00`/`11→00`）：若前站 defect 對後站 noise 有 ghosting 影響
- 疊加類（`10→11`/`01→11`）：若同位置疊加 defect 在實機真的出現

#### 10.2.3 架構升級
- Siamese twin encoder（前後站各一 encoder、decoder 端融合）
- Cross-attention 對齊 feature
- Deformable / correlation 處理輕微 misalignment

---

## 11. 與 Background_Removal_Net 的關係

NC_PSF_Net **直接共用** Background_Removal_Net 的：
- `generate_psf.py`（PSF 生成）
- `gaussian.py`（局部 inpaint 工具）
- `loss.py`（Focal Loss）
- `defects/*.yaml`（PSF configs）

**重新撰寫**的部分（v2）：
- `dataloader.py`：4 通道配對 + 9 種 case 合成
- `model.py`：`in_channels=4`
- `trainer.py`：通道調整
- `inference.py` / `eval_synthetic.py`：4 通道輸入，5-panel 視覺化（去掉 R2）

核心邏輯：原專案的 die-to-die 黃金法則與跨站對比合併為一條精確判斷規則（`next_T=1 AND next_R=0 AND prev_T=0`），並透過「後站 T-only 家族」**四錨點**（`00→10`、`01→10`、`10→10`、`11→10`）強制覆蓋四種「後站站內看都是 T-only」的歧義情境，逼模型學到「只看 `prev_T`、忽略 `prev_R`」。
