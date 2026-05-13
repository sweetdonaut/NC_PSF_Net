# NC_PSF_Net 設計文件 — Hybrid（當前實作目標）

> **本文件 status**：當前採用設計。**狀態：規劃完成，code 尚未實作**。  
> 目前 code 仍對應 `archive/design_strategyB.md`（純合成 baseline）。Hybrid 實作後本文件成為 code 的依據。
>
> **核心定位**：Hybrid = 真實前後站影像（策略 A 的底圖）+ 沿用策略 B 的全部 inpaint 機制。**不是「折衷」，是「真正能落地的策略 A」**——純策略 A 因強制四錨點機制湊不齊而工程上失敗（詳見 `archive/design_strategyA.md`）。
>
> **相對於策略 B 唯一改變的元件**：底圖來源。其餘（9 種 case 設計、強制四錨點機制、模型架構、loss、訓練流程）完全沿用。
>
> **命名規則**：用 `前pattern→後pattern` 當 case 代號（例如 `00→10`），pattern 為 2 位元 `(T, R)`，`10` 表示 T die 有 defect、R die 沒有。

---

## 1. 為什麼是 Hybrid

策略 A（純真實底圖）跟策略 B（純合成）各有不能放棄的優勢：

| 來自策略 A | 來自策略 B |
|---|---|
| 真實前站 defect 外觀分布 | 強制四錨點機制可控 |
| 真實跨站 noise 結構 | 9 種 case 分布可控 |
| 持續類 case 自然存在於底圖 | inpaint 配置每次都不同 → 無限資料增強 |
| 訓練 distribution 對齊 inference | GT 100% 來自已知 inpaint 位置 |

純策略 A 的問題：底圖中持續類 case 的密度由「真實前站 defect 多少」決定，不可控。多數 patch 沒有 `10→10` 或 `11→10` 樣本，強制四錨點機制無法執行 → 模型可能發展 shortcut（詳見 `archive/design_strategyA.md` §2）。

→ **Hybrid 的設計選擇**：用真實底圖取得 A 的所有優勢，仍 inpaint 全部四錨點取得 B 的訓練可控性，接受極低機率的「inpaint 撞到底圖既有 defect」雜訊。

---

## 2. 規格摘要（相對於 strategy B 的變動）

| 項目 | Strategy B（已 archive） | Hybrid（本文件） |
|---|---|---|
| 底圖來源 | BRN grayscale → process shift 合成 | **真實前後站影像對** |
| 跨站 noise 來源 | 合成 brightness/offset/Gaussian | **真實製程造成** |
| 「乾淨後站」篩選 | 不需要（底圖天生乾淨） | **機台篩選**（使用者已確認可行） |
| 真實前站 defect | 沒有 | **自然存在於底圖** |
| 持續類 (`10→10` 等) | 全靠 inpaint | **底圖自然 + inpaint 補強同時並存** |
| 強制四錨點 | 每 patch 全 inpaint | **每 patch 全 inpaint**（不變） |
| 9 種訓練 case | 全 inpaint | **全 inpaint**（不變） |
| Inpaint 邏輯 | 隨機 patch 位置 | **隨機 patch 位置**（不變） |
| 避撞底圖 defect | 不需要 | **第一版不做**，附註其他方案 |
| GT mask | inpaint 位置 | **inpaint 位置**（不變） |
| Model / Loss / Trainer | – | **完全不變** |

→ **真正改變的只有底圖來源**。

---

## 3. 背景與動機

晶片製造採多站點流程。每個站點可能產生 defects：

- **前站**：先一步處理，可能已有 defects
- **後站**：在前站之後處理，可能產生新的 defects

後站影像會同時包含前站既有 defects（自然延續）與後站新增 defects。我們**只關心後站新增**的部分，但前站延續 defects 對 rule-based 算法構成大量干擾。

核心策略：**讓 U-Net 同時看前後站影像，學會對比兩者，只標出後站獨有的 PSF defect**。

---

## 4. 資料結構

### 4.1 die-to-die 影像規格

每張前/後站影像是 **2 通道灰階**（die-to-die comparison）：

- `target` (T)：目標 die
- `ref` (R)：相鄰參考 die

前後站對應同一片晶圓的同一位置，**前後站之間像素對齊**（MVP 假設）。

模型輸入 = 前後站 concat = **4 通道**：

```
[prev_T, prev_R, next_T, next_R]
```

### 4.2 die-to-die 黃金法則

對單站二通道：**只有 T-only (`10`) pattern 才是 anomaly 候選**。

- `10` (T-only) → anomaly 候選
- `01` (R-only) → ref die 缺陷，**不算**
- `11` (T+R) → 兩 die 同位置都有，可能是 wafer 級或巧合，**不算**

### 4.3 模型要學的精確判斷規則

合併「站內 die-to-die 黃金法則」與「跨站對比」後，「該標的 anomaly」可以寫成一個精確判斷式：

> **anomaly = (`next_T = 1`) AND (`next_R = 0`) AND (`prev_T = 0`)**

**重點：`prev_R` 不在判斷規則中**。R 通道在前站的狀態與「後站 T 上是否為新 anomaly」邏輯上無關。

模型要學會的不是「找後站獨有」，而是「**判斷後站站內 T-only 是否在前站 T 通道也存在**」。R 通道（前後站皆然）提供 die-to-die 比對的視覺特徵，但不直接決定標不標。

### 4.4 data/ 目錄與檔案格式

```
data/
├── train/
│   ├── prev/   ← 2 通道 tiff（每張 shape: (2, H, W) CHW float32）
│   └── next/   ← 2 通道 tiff（同上）
├── val/
│   ├── prev/
│   └── next/
├── test/
│   ├── prev/
│   └── next/
└── manifest.json   ← 紀錄合成 / 配對參數以供重現
```

**檔案格式細節**：

- 每張影像 2 通道灰階，存成 CHW float32 tiff
- 通道 0 = target die，通道 1 = reference die
- `prev/<name>.tiff` ↔ `next/<name>.tiff` 同名配對（同一片晶圓同一位置）

**重要：`data/` 裡完全沒有 GT mask 檔**。GT 的來源見 §4.6。

### 4.5 影像內容定義

「乾淨」精確定義 = **沒有後站新增 defect**；前站延續類 defect **可以存在**（這是真實生產的自然狀態）。

| Phase | prev 影像內容 | next 影像內容 |
|---|---|---|
| **Phase 1**（目前實作）| BRN grayscale + 微弱 die noise，**完全沒有任何 defect** | 同一張 grayscale + process shift + die noise，**完全沒有任何 defect** |
| **Phase 2**（未來）| **真實前站影像**，含真實前站既有 defects | **真實「乾淨」後站影像**，**含前站延續到後站的真實 defects** |

Phase 2 的「乾淨」判斷由機台篩選，定義見 §9。

### 4.6 GT 動態合成機制

GT 不存於 `data/`，而是**每次 `__getitem__` 被呼叫時動態合成**。

`dataloader.py::generate_paired_defects()` 流程：

1. 載入該 patch 的前後站 4 通道
2. 50% 機率不放 defect → 回傳 GT 全 0
3. 否則：在前後站通道上 inpaint 一組合成 PSF defects（按 9 種 case + 強制四錨點，見 §6）
4. 把「屬於正樣本 case（`00→10`、`01→10`）」的 inpaint 位置標進 GT mask
5. 回傳 inpaint 後的 4 通道 + 對應的 GT mask

→ 同一張底圖在不同 epoch 看到的 GT **不同**（因為 inpaint 隨機）。
→ Val/Test 用固定 random seed 來重現一致的合成 GT，作為跨 epoch 比較訊號（`trainer.py::evaluate_synthetic`）。

### 4.7 為什麼這樣設計

#### 4.7.1 為什麼底圖只放「乾淨」影像（不含後站新 anomaly）？

如果底圖裡有「位置未知」的真實後站 anomaly：
- 該位置真實是 anomaly（GT 應為 1）
- 但 dataloader 不會 inpaint 那裡，所以 GT mask 標 0
- model 看到「視覺上是 anomaly + 訓練標籤 = 0」→ 學到「這種 defect 不該標」
- → 訓練毒藥，recall 受傷

「乾淨」假設避開這個問題。**前站延續類**（`10→10`、`01→01`、`11→11`）允許存在於底圖，因為它們本來就 GT=0，跟「未 inpaint 區域 GT=0」一致，不會誤導 model。

#### 4.7.2 為什麼沒有 GT mask 檔，而是動態合成？

- **沒有真實標註可用**：人工標註 pixel-level wafer defect 成本極高，本專案目前不靠人工標註
- **無限資料增強**：每次 `__getitem__` 用不同的 defect 位置 / 強度 / 形狀 / case 分配 → 等效樣本量遠大於 300 張底圖

預先存固定 GT 會把這兩個優勢都丟掉。

#### 4.7.3 為什麼 train/val/test 都用同樣的動態合成？

- **一致性**：所有 set 用同樣的判斷規則，沒有 train-eval 偏差
- **可比較性**：跨 epoch 監控 val AUROC 是有意義的訓練收斂訊號（seed 鎖定使每個 epoch 的 val 合成幾乎相同）
- **覆蓋率可控**：強制四錨點機制保證每個 patch 都覆蓋四個關鍵 case，靜態 GT 做不到

#### 4.7.4 為什麼前後站像素對齊（MVP 假設）

判斷規則需要逐 pixel 比較 `prev_T` 與 `next_T`。若像素不對齊，model 必須同時學「跨站配準」+「找新 defect」兩件事。MVP 直接假設對齊；Phase 2 的真實資料若無法對齊，需要前置配準（見 §12.1 第 1 點）。

### 4.8 這個設計能驗證什麼、不能驗證什麼

| ✓ 能驗證 | ✗ 不能驗證 |
|---|---|
| Pipeline 邏輯正確（dataloader、model、loss 沒 bug） | Model 對**真實**後站新 defect 的偵測能力 |
| Model 能擬合合成 defect distribution | 合成 PSF 與真實 defect 的 domain gap 影響 |
| 9 種 case 都被訓練到（透過 case coverage 統計） | 真實 noise / 紋理變化下的 robustness（Phase 1） |
| Phase 2 多驗證：model 能在真實 noise 下處理合成 defect、能正確忽略真實前站 defect | Phase 2 仍不衡量真實 anomaly 偵測效能 |

### 4.9 本質：「合成監督學習」

很多人看到「沒有 GT 檔」會以為這是 unsupervised。**不是**。

這是 **supervised learning，只是監督訊號由 dataloader 動態生成而非預先存檔**：

- `focal_loss(pred, dynamic_GT)` 仍是 supervised
- GT 來自我們已知位置的 inpaint，100% 可信
- 比真實 anomaly 標註便宜（不用人標）
- 比 unsupervised 有明確 signal（明確知道哪些 pixel 該標 1）

**代價**：合成 distribution ≠ 真實 distribution，所以「合成 AUROC = 1.0」**不能保證真實效能**。Phase 2 用真實底圖能稍微縮小 gap（model 至少看到真實 noise），但 GT 仍是合成 inpaint，無法完全消除這個落差。

一句話總結整套設計：

> **用「乾淨底圖 + 動態合成 inpaint」當訓練資料，犧牲「真實 anomaly 效能驗證」這個無法負擔的成本，換取「無限合成訓練樣本 + 完美可控 GT + 100% case coverage」。**

---

## 5. Case 分析：Hybrid 下的雙來源

### 5.1 判斷框架

訓練影像 = **底圖 + 主動 inpaint 的 defects**。Training signal 有兩種來源：

- **來源 A（底圖隱含）**：底圖本身自然產生這個 case → 模型透過大量背景像素自動學到
- **來源 B（必須 inpaint）**：底圖永遠不會出現 → 必須合成才能訓練

Hybrid 跟策略 B 的差別 = 底圖能提供多少 case：
- 策略 B 底圖只提供 `00→00`
- **Hybrid 底圖提供 `00→00` + 持續類 + 部分移除類**（真實生產的自然分布）

### 5.2 Hybrid 下「底圖」的定義

底圖 = 真實前站影像 + 機台篩選的「乾淨」後站影像。

「乾淨」精確定義為「**沒有後站新增 defect**」，但前站既有 defects 仍自然延續到後站影像（這是真實生產狀態）。

→ Hybrid 底圖自然產生：
- `00→00`（背景大部分像素）
- `10→10`, `01→01`, `11→11`（前站 defect 延續）
- 少量 `11→10`, `11→01`（製程選擇性移除）
- 少量 `10→00`, `01→00`, `11→00`（製程整體移除）

→ 底圖**不會自然產生**（仍需 inpaint）：
- `00→10`（GT=1 正樣本）
- `01→10`（GT=1 正樣本）
- `00→01`, `00→11`（後站新增類）

### 5.3 16 種 case 完整評估表

| Pattern | 製程情況 | inference 遇到？ | 底圖自然產生？ | GT | Hybrid 處理 |
|---|---|---|---|---|---|
| `00→00` | 正常背景區 | 是（大部分像素） | ✅ 是 | 0 | 不必 inpaint |
| `10→10` | 前站 T-only 缺陷往後傳遞 | 是 | **✅ 真實前站自然延續** | 0 | **底圖隱含 + inpaint 補強（錨點）** |
| `01→01` | 前站 R-only 缺陷往後傳遞 | 是 | ✅ 真實延續 | 0 | 底圖隱含 + inpaint 補強 |
| `11→11` | 前站 T+R 缺陷往後傳遞 | 是 | ✅ 真實延續 | 0 | 底圖隱含 + inpaint 補強 |
| `00→10` | **後站新增 T-only 缺陷（要找的 anomaly）** | 是（**目標！**） | ❌ 否 | **1** | **必須 inpaint** ⭐ |
| `00→01` | 後站新增 R-only 缺陷 | 是 | ❌ 否 | 0 | 必須 inpaint |
| `00→11` | 後站新增 T+R 缺陷 | 是 | ❌ 否 | 0 | 必須 inpaint |
| `10→00` | 前站 T-only 被製程移除 | 罕見 | △ 可能有 | 0 | 不必 inpaint |
| `01→00` | 前站 R-only 被製程移除 | 罕見 | △ 可能有 | 0 | 不必 inpaint |
| `11→00` | 前站 T+R 被製程整體移除 | 罕見 | △ 可能有 | 0 | 不必 inpaint |
| `10→01` | 前站 T 缺陷消失 + 後站 R 新增 | 罕見 | ❌ 否 | 0 | 不必 inpaint（與 `00→01` 等價） |
| `01→10` | **前站 R 缺陷消失 + 後站 T 新增 anomaly** | 罕見 | ❌ 否 | **1** | **必須 inpaint** ⭐ |
| `10→11` | 前站 T-only + 後站 R 又新增 | 極罕見 | ❌ 否 | 0 | 不必 inpaint |
| `01→11` | 前站 R-only + 後站 T 又新增 | 極罕見 | ❌ 否 | 0 | 不必 inpaint |
| `11→10` | 前站 T+R，後站 R 被製程選擇性移除 | 是 | **△ 可能有** | 0 | **底圖可能隱含 + inpaint 補強（錨點）** ⚠️ |
| `11→01` | 前站 T+R，後站 T 被製程選擇性移除 | 是 | △ 可能有 | 0 | 底圖可能隱含 + inpaint 補強 |

### 5.4 為什麼錨點仍要 inpaint（不只靠底圖）

雖然 hybrid 底圖自然包含持續類，**強制四錨點仍須 inpaint**，理由：

1. **底圖密度不可控**：300 張影像中真實前站 defect 數量假設每張 5 個 → 約 1500 個事件。切成 patch 後平均每 patch 只有 0.3 個持續類事件 → **多數 patch 缺乏錨點訓練 signal**
2. **`11→10` 在底圖中極稀疏**：製程選擇性移除是中等頻率事件，patch 級幾乎遇不到
3. **失去四錨點對等性會誘發 shortcut**：模型在「只有 `00→10` 和 `01→10` 兩個 patch 內錨點」的情境下，可能學成「跨站對比後站獨有就標 1」，忽略 `prev_T` 的關鍵作用

→ Hybrid 的真實優勢不是「省 inpaint capacity」，而是「**底圖提供額外的真實 reference signal 作為隱性負樣本**」，inpaint 仍然執行完整錨點機制。

---

## 6. 最終 9 種訓練 case（與策略 B 完全相同）

| Pattern | 製程情況 | GT | 訓練角色 | 來源 |
|---|---|---|---|---|
| `10→10` | 前站 T-only 缺陷往後傳遞 | 0 | 後站 T-only 家族錨點（前 `10`） | 底圖 + inpaint |
| `01→01` | 前站 R-only 缺陷往後傳遞 | 0 | R-only 延續分布覆蓋 | 底圖 + inpaint |
| `11→11` | 前站 T+R 缺陷往後傳遞 | 0 | T+R 延續分布覆蓋 | 底圖 + inpaint |
| `00→10` | 後站新增 T-only 缺陷 | **1** ⭐ | 後站 T-only 家族錨點（前 `00`，正樣本 #1） | **僅 inpaint** |
| `01→10` | 前站 R 缺陷消失 + 後站 T 新增 | **1** ⭐ | 後站 T-only 家族錨點（前 `01`，正樣本 #2） | **僅 inpaint** |
| `00→01` | 後站新增 R-only 缺陷 | 0 | 後站新 R-only | 僅 inpaint |
| `00→11` | 後站新增 T+R 缺陷 | 0 | 防站內 T 誘惑 | 僅 inpaint |
| `11→10` | 前站 T+R，後站 R 缺陷被製程移除 | 0 | 後站 T-only 家族錨點（前 `11`） | 底圖（稀疏）+ inpaint |
| `11→01` | 前站 T+R，後站 T 缺陷被製程移除 | 0 | T+R 部分移除分布覆蓋 | 底圖（稀疏）+ inpaint |

### 6.1 強制機制：「後站 T-only 家族」四錨點

**每個 patch 至少各放 1 個 `00→10`、`01→10`、`10→10`、`11→10`**（與策略 B 完全相同）。

四個 case 在「後站站內看」全部都是 T-only，差別只在前站 pattern：

| Pattern | prev (T R) | GT | 意義 |
|---|---|---|---|
| `00→10` | 0 0 | **1** | 真新增（前站全乾淨） |
| `01→10` | 0 1 | **1** | 真新增（前站只有 R 干擾） |
| `10→10` | 1 0 | 0 | 假新增（前站 T 就有） |
| `11→10` | 1 1 | 0 | 假新增（前站 T+R 都有） |

**唯一能正確區分的訊號是 `prev_T`**。四錨點齊全才能逼模型學到「只看 `prev_T`，忽略 `prev_R`」的正確規則。

### 6.2 合成流程

每個 patch：

1. **50% 機率完全不放 defect** → GT 全 0
   - 注意：即使不放 inpaint，底圖仍提供真實前站 defect 的負樣本訓練 signal
2. 否則：
   - 隨機 N 個 defects（範圍預設 4~10）
   - **強制塞 1 個 `00→10` + 1 個 `01→10` + 1 個 `10→10` + 1 個 `11→10`**
   - 剩下 N-4 個從 9 種 case 均勻隨機抽

---

## 7. 模型架構

### 7.1 沿用策略 B 的 U-Net 骨幹（完全不變）

```python
SegmentationNetwork(in_channels=4, out_channels=2)
```

- Background_Removal_Net 同款 encoder-decoder + SPPF + SEBlock
- 第一層 conv 接受 4 通道輸入

### 7.2 為什麼不升級架構

- 真實底圖 + 完整 inpaint 機制下，4 通道 concat 讓 encoder 自己學跨站對比已足夠
- Hybrid 的訓練 distribution 比策略 B 更貼近 inference，先驗證真實效能再考慮升級

---

## 8. 訓練策略

完全沿用策略 B：

- **Loss**：Focal Loss + cosine gamma schedule，`alpha=0.75`
- **資料增強**：patch 切割 + 隨機 inpaint 配置（每次 `__getitem__` 不同）
- **資料切分**：300 張 → 240 train / 30 val / 30 test (8:1:1)
- **val/test GT**：合成 inpaint GT（與策略 B 一致，用於監控訓練收斂）

---

## 9. 「乾淨後站」篩選機制

### 9.1 機制設計

使用者確認**機台可篩選**「沒有後站新增 defect」的影像。具體機制由生產線負責，本專案接收已篩選過的資料。

### 9.2 篩選品質的影響

「乾淨後站」品質直接決定 hybrid GT 純淨度：

- **理想情況**：篩選 100% 準確 → GT 純淨，模型訓練無雜訊
- **漏網情況**：未檢出的真實 `00→10` 留在底圖中 → 該位置 GT=0（我們不知道它是 anomaly），模型學到「這種視覺特徵不該標」→ recall 受傷

機台篩選失效的常見來源：
- 邊界 case 介於「乾淨」和「有 defect」之間
- 篩選 threshold 過鬆（誤判為乾淨）
- 機台只看 target 通道、忽略 ref 通道的訊號

### 9.3 緩解措施

第一版 Hybrid 完全信任機台篩選結果。若實機驗證發現 recall 異常下降，可考慮：

1. **多輪篩選**：機台篩選後再用 strategy B 訓練的初版模型粗篩，雙重把關
2. **人工抽檢**：對篩選通過的影像抽檢，估算漏網率
3. **Loss 對稱性檢查**：訓練後分析 model 在不同訓練 patch 上的 loss 分布，找出疑似有未標 anomaly 的 patches

---

## 10. Inpaint 避撞策略

### 10.1 第一版設計：完全不做避撞

Inpaint 位置完全隨機，接受偶爾撞到底圖既有真實 defect 的 GT 雜訊。

**撞到的後果**（最壞 case：在底圖 `10→10` 位置上 inpaint `00→10`）：
- inpaint 強度疊加在底圖 next_T 既有 defect 上
- GT 標 1（按 inpaint）
- 但底圖 `prev_T` 實際有 defect → 按真實判斷規則 GT 應為 0
- → **該位置 GT 錯標**

**撞到機率估算**：
- 真實前站 defect 佔底圖 ~0.01-0.1% pixel
- 每 patch 約 4-10 個 inpaint，累積撞到機率 ~0.04-1%
- 訓練集錯標比例 ~0.04-1%，對訓練影響可忽略

### 10.2 附註：未來瓶頸時的避撞方案

若 model 在真實 inference 上發現某些位置 systematic 誤判（heatmap 與底圖既有 defect 位置高度相關），可逐步升級：

- **方案 2（threshold 避撞）**：inpaint 前粗略檢查 `prev_T` 該位置強度，超過 threshold 跳到下個隨機位置。簡單，但 threshold 調參敏感
- **方案 3（detector 避撞）**：用前置 detector（rule-based 或 strategy B 訓練的初版模型）標出底圖 defect mask，inpaint 嚴格避開。乾淨但複雜，detector 誤差會傳播
- **方案 4（GT 矯正）**：inpaint 後重新計算正確 GT，需要知道底圖 defect 位置 + 強度。最嚴謹但實作門檻最高

---

## 11. PSF Defect 生成（沿用策略 B）

直接共用 `Background_Removal_Net/src_core/generate_psf.py`：

- 環形光瞳 + Zernike 像差 → FFT → Poisson/Gaussian noise → connected-peak 清理
- 支援純量 PSF 與 Richards-Wolf 向量 PSF
- 預先用 multiprocess pool 產生 PSF defect pool

**Vector PSF caveat**（沿用 strategy B 經驗）：vector PSF 在合成資料上 intensity 是「pixel-perturbation magnitude」（不是真實光學 amplitude），建議使用 `type4_vector_strong.yaml`（intensity 60-80）而非原 `type4_vector.yaml`（intensity [[8, 12]]）。Hybrid 因為底圖為真實影像，**intensity 校準需要重新估算**（真實影像的動態範圍可能跟合成不同）。

---

## 12. 已知限制與後續方向

### 12.1 已知限制

1. **像素對齊假設**：MVP 假設前後站像素級對齊，實際可能需要預配準
2. **真實 PSF defect 仍存在 domain gap**：合成 PSF 與後站新出現的真實 defect 外觀可能不同
3. **資料量**：300 張底圖偏少，真實前站 defect 多樣性是 hybrid 效能的上限
4. **乾淨後站篩選品質**：未檢出的真實 `00→10` 變訓練毒藥（§9.2）
5. **GT 撞底圖機率**：估算 < 1%，但若實機有系統性問題需要升級避撞方案
6. **真實效能未驗證**：合成 GT 上的 AUROC 不等於真實場景效能

### 12.2 後續方向

#### 12.2.1 真實標註資料驗證

收集人工標註的小驗證集，量化 hybrid 對真實 inference 的效能（vs strategy B baseline）。

#### 12.2.2 加入未訓練 case

若實機驗證發現某些 case 表現不穩，可考慮加入：
- 消失類（`10→00`/`01→00`/`11→00`）：若前站 defect 對後站 noise 有 ghosting 影響
- 疊加類（`10→11`/`01→11`）：若同位置疊加 defect 在實機真的出現

#### 12.2.3 架構升級

- Siamese twin encoder（前後站各一 encoder、decoder 端融合）
- Cross-attention 對齊 feature
- Deformable / correlation 處理輕微 misalignment

#### 12.2.4 避撞升級

若 §10.1 的 < 1% 雜訊在實機顯現出問題，依序評估 §10.2 的方案 2/3/4。

---

## 13. 實作 checklist（從目前 code 演化到 hybrid）

> **重要釐清**：目前 code 仍對應 `archive/design_v1_3ch.md` 的 **3 通道版本**（commit `b892a97`）。策略 B 的 2 通道設計文件雖然存在於 archive，但**從未實作為 code**。因此從現有 code 到 hybrid，需要同時完成「3 通道 → 2 通道重構」與「合成底圖 → 真實底圖」兩件事。

### 13.1 從現有 code (commit `b892a97`) 到 hybrid 的完整改動

| 元件 | 改動內容 | 工作量 |
|---|---|---|
| `src_core/model.py` | `SegmentationNetwork(in_channels=6, ...)` → `in_channels=4` | 一行 |
| `src_core/dataloader.py` | `CASES` dict：14 條目 (A1-A7, B1-B7) → 9 條目（pattern 命名）<br>`CHANNEL_ORDER`：6 名 → 4 名<br>`generate_paired_defects`：3 通道 patch → 2 通道 patch<br>`_assign_cases`：強制 A1+B1 → 強制四錨點（`00→10`/`01→10`/`10→10`/`11→10`）<br>讀入的底圖換成真實前後站影像（路徑沿用 `prev_path`/`next_path`） | **大幅改動** |
| `src_core/trainer.py` | 輸入鍵名不變 (`paired_input`)、`num_defects_range` 下限調整為 4 | 小調整 |
| `src_core/inference.py` | sliding-window stack：6 通道 → 4 通道；視覺化 panel：7-panel → 5-panel | 中等 |
| `src_core/eval_synthetic.py` | 同 inference.py，stack 與視覺化通道數調整 | 中等 |
| `scripts/build_paired_dataset.py` | **完全重寫**：合成 → 接收真實前後站影像對 + 機台篩選結果排版到 train/val/test | 重寫 |
| `src_core/loss.py` | **完全不變** | – |
| `src_core/defects/*.yaml` | **完全不變** | – |

### 13.2 哪些核心邏輯確實不變

雖然 channel 數涉及多個檔案，但以下核心算法邏輯**完全沿用**：

- 模型架構：UNet + SPPF + SEBlock 的 encoder-decoder 骨幹
- Loss：Focal Loss + cosine gamma schedule
- 訓練流程：sliding-window 訓練、AUROC 評估、checkpoint 儲存
- Inference 演算法：sliding-window + center-crop stitching
- PSF defect 生成：`generate_psf.py` 完全不變，pool 機制不變
- `_create_one_defect`、`apply_local_defect_to_background` 等局部工具完全不變

### 13.3 視覺化檔案

- 保留現有 `docs/figures/strategyB_*.png`（其實是 3 通道版本的真實實驗結果，命名因為已 commit 不另外改）
- Hybrid 訓練後加入 `hybrid_*.png` 視覺化

---

## 14. 與策略 A、策略 B 的關係（一頁總結）

| 維度 | 純策略 A | 純策略 B | **Hybrid（本文件）** |
|---|---|---|---|
| 底圖 | 真實前後站 | 合成乾淨 | **真實前後站** |
| 持續類訓練 | 只靠底圖（密度不可控）| 全 inpaint | **底圖 + inpaint 雙重來源** |
| 強制四錨點 | 無法保證 | 完整保證 | **完整保證** |
| GT 純淨度 | 受篩選品質影響 | 100% | **受篩選品質影響** |
| 實作可行性 | 不可行 | 已驗證 | **規劃完成** |
| 真實 inference 對齊 | 高 | 低 | **高** |
| 文件 | `archive/design_strategyA.md` | `archive/design_strategyB.md` | **本文件** |
