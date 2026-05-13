# NC_PSF_Net 設計文件（Refresh 版）

> **這個文件的目的**：給第一次接觸這個專案的人一條完整、線性的閱讀路徑，把「為什麼這樣設計」講清楚。
>
> **跟其他文件的關係**：
> - `design_hybrid_overview.md` 是 3 分鐘版（決定要不要深入）
> - 本文件是完整版（決定深入後讀什麼）
> - `design_hybrid.md` 是歷史版本，正在被本文件取代
> - `archive/` 是被放棄的路線（v1 三通道、純策略 A、純策略 B），詳見 [§A. Archive 指南](#a-archive-指南)
>
> **狀態**：Phase 1 已實作（commit `bace3b2`），Phase 2 等真實資料到位。詳見 [§11. 實作 status](#11-實作-status)。

---

## 目錄

1. [一頁理解這個專案](#1-一頁理解這個專案)
2. [背景：問題是什麼](#2-背景問題是什麼)
3. [核心判斷規則](#3-核心判斷規則)
4. [為什麼是 Hybrid（三條策略的演化）](#4-為什麼是-hybrid三條策略的演化)
5. [訓練 Case 設計](#5-訓練-case-設計)
6. [合成監督的本質](#6-合成監督的本質)
7. [資料 Pipeline](#7-資料-pipeline)
8. [Phase 1 vs Phase 2](#8-phase-1-vs-phase-2)
9. [模型與訓練](#9-模型與訓練)
10. [已知限制與後續方向](#10-已知限制與後續方向)
11. [實作 status](#11-實作-status)
12. [附錄](#附錄)

---

## 1. 一頁理解這個專案

**做什麼**：晶圓檢測，多站點製程。輸入「前站、後站」同位置影像對，輸出後站**新增** PSF defect 的 pixel-level heatmap。前站延續來的 defect 要忽略。

**怎麼做**：4 通道輸入 U-Net 學習以下判斷規則：

```
anomaly = (next_T == 1) AND (next_R == 0) AND (prev_T == 0)
```

> `T` = target die，`R` = reference die（die-to-die 比對的另一顆 die）；前站 / 後站各 2 通道，總共 4 通道。`prev_R` 不在規則中——學會這個非對稱性是模型的核心任務。

**為什麼 Hybrid**：純真實底圖（策略 A）湊不齊四錨點訓練樣本；純合成底圖（策略 B）跟真實 inference 分布有 gap。**Hybrid = 真實底圖（取得真實 noise + 真實前站 defect 外觀）+ dataloader 動態 inpaint（取得可控的 case 覆蓋與 GT）**。

**狀態**：
- **Phase 1**：2 通道 pipeline + 9 case + 四錨點機制就位，跟合成底圖訓練。val/test AUROC = 1.0000。已完成（commit `bace3b2`）。
- **Phase 2**：底圖切換為真實前後站影像。等真實資料 + 機台篩選結果到位。只需重寫 `scripts/build_paired_dataset.py` 或者使用者手動排版資料。

**重要認知**：「合成 AUROC = 1.0」**不等於**真實效能。詳見 [§6.4](#64-能驗證-不能驗證的) 跟 [§10](#10-已知限制與後續方向)。

---

## 2. 背景：問題是什麼

### 2.1 多站點製程的 defect 偵測

晶片製造採多站點流程。每個站點可能產生 defects：

- **前站**：先一步處理，可能已產生一些 defects（不是我們要找的目標）
- **後站**：在前站之後處理，可能在前站影像基礎上產生新的 defects

後站影像會同時包含：
1. **前站既有 defects 的延續**（拍後站時前站的 defect 自然還在那）
2. **後站新增 defects**（要找的目標）

**核心任務**：只標出 (2)，忽略 (1)。

純 single-station rule-based 算法無法區分（兩者在後站影像上視覺特徵類似）。這就是為什麼需要 cross-station 比對 + ML model。

### 2.2 die-to-die 比對的標準假設

晶圓上有重複的 die（晶粒）。常見的 defect 偵測技術是 **die-to-die 比對**：把同一片晶圓上**相鄰兩個 die** 的同位置 pixel 比較，挑出「只在 target die 上有、ref die 沒有」的訊號當 defect 候選。

每個站點的影像因此是 **2 通道**：
- `T` (target)：目標 die
- `R` (reference)：相鄰參考 die

對單站二通道，**站內 die-to-die 黃金法則**：

| Pattern (T, R) | 解讀 | 算 anomaly？ |
|---|---|---|
| `00` | 兩 die 都乾淨 | 否（背景） |
| `10` (T-only) | 只在 target die 有訊號 | **候選** |
| `01` (R-only) | 只在 ref die 有訊號 | 否（缺陷在 ref，不在 target） |
| `11` (T+R) | 兩 die 同位置都有 | 否（可能 wafer 級重複圖案或巧合） |

→ 只有 `T-only`（`10`）是站內 anomaly 候選。

### 2.3 為什麼需要 cross-station

單看後站站內找 `10` 候選不夠——「前站延續類」的 defect 在後站也是 `T-only` pattern，會混進候選清單。

→ 加入前站影像做 cross-station 比對：如果同位置在前站 T 也是 `1`，那它是延續來的，不算。

這就需要 4 通道輸入 `[prev_T, prev_R, next_T, next_R]` 加 U-Net 學習組合判斷規則。

---

## 3. 核心判斷規則

### 3.1 規則本身

從「站內 die-to-die 黃金法則」（[§2.2](#22-die-to-die-比對的標準假設)）擴展到跨站：

> **anomaly = (`next_T == 1`) AND (`next_R == 0`) AND (`prev_T == 0`)**

逐條件解讀：

| 條件 | 意義 |
|---|---|
| `next_T == 1` | 後站 target die 有訊號（站內候選） |
| `next_R == 0` | 後站 reference die 同位置沒訊號（排除 wafer 級重複圖案） |
| `prev_T == 0` | 前站 target die 同位置沒訊號（排除前站延續類） |

### 3.2 為什麼 `prev_R` 不在規則中

「前站 R die 同位置有沒有訊號」跟「後站 T 是否為新 anomaly」**邏輯上無關**：

- 若後站 T 真的新增 defect，前站 R 是 0 還是 1 都不影響「新增」的事實
- 若後站 T 是延續類，由 `prev_T == 1` 排除，跟 `prev_R` 也無關

但 `prev_R` 仍是模型 input，**提供前站 die-to-die 對比的視覺特徵**。它的角色是「視覺輔助」不是「決定變數」。

### 3.3 模型要學的非對稱性

這個規則隱含一個**訓練上的非對稱性**：

- `prev_T` 跟 `next_T` 在規則中**同等重要**（兩者必須同時滿足才標 1）
- `prev_R` 跟 `next_R` **角色不對稱**：`next_R` 在規則裡，`prev_R` 不在

模型必須學會「看 4 個 channel 但只用其中 3 個做決定」。這不是顯然的，必須透過訓練 case 設計來逼出——細節見 [§5.3 四錨點機制](#53-四錨點機制重點)。

---

## 4. 為什麼是 Hybrid（三條策略的演化）

### 4.1 演化脈絡

```
v1 (commit b892a97)
 │   最初的 MVP 實作。
 │
 ├─→ 策略 A（純真實底圖）→ 四錨點密度不可控，不可行（§4.2）
 │
 ├─→ 策略 B（純合成底圖）→ 設計上可行但 distribution gap 太大；從未實作 code（§4.3）
 │
 └─→ Hybrid（真實底圖 + 動態 inpaint，當前路線）
        ├─ Phase 1：用合成底圖實作 hybrid pipeline（commit bace3b2）
        └─ Phase 2：切換到真實底圖（等資料到位）
```

詳細歷史筆記在 [`archive/`](#a-archive-指南)。

### 4.2 純策略 A 的問題：四錨點密度不可控

策略 A：用真實前後站影像當訓練資料，期待真實 defect 提供訓練 signal。

問題：**強制四錨點機制**（[§5.3](#53-四錨點機制重點)）需要每個訓練 patch 都看到 `00→10` / `01→10` / `10→10` / `11→10` 四種視覺相似但 GT 不同的 case。

- `10→10`、`11→10` 來自「真實前站 defect 延續到後站」，但真實前站 defect 在底圖中密度約 0.01-0.1% pixel
- 切成 128² patch 後，**大多數 patch 沒有任何持續類 defect**
- → 四錨點根本湊不齊 → 模型可能發展 shortcut（[§5.4](#54-shortcut-風險與防範)）

策略 A 在「四錨點訓練」這件事上工程上不可行。

### 4.3 純策略 B 的問題：distribution gap

策略 B：底圖完全合成（BRN grayscale + 合成 noise + 合成 process shift），所有訓練 case 都用 inpaint 產生。

優勢：
- 強制四錨點 100% 可控（每個 patch 都按設計放置）
- 9 種 case 分布 100% 可控
- GT 100% 來自已知 inpaint 位置，不會誤標

劣勢：**合成 distribution ≠ 真實 inference distribution**
- 真實晶圓的 noise 結構（光學紋理、感測器雜訊、製程隨機性）合成 noise 模擬不出來
- 真實前站 defect 的外觀分布（形狀、強度、位置 prior）合成不出來
- → 模型在合成 AUROC 高，真實 inference 可能掉很多

### 4.4 Hybrid 的選擇

| 來自策略 A 的優勢 | 來自策略 B 的優勢 |
|---|---|
| 真實前站 defect 外觀分布 | 強制四錨點 100% 可控 |
| 真實跨站 noise 結構 | 9 種 case 分布可控 |
| 持續類 case 自然存在於底圖 | 每次 inpaint 配置不同 → 無限資料增強 |
| 訓練 distribution 對齊 inference | GT 100% 來自已知 inpaint 位置 |

**Hybrid 的設計選擇**：

1. **底圖**用真實前後站影像 → 取得策略 A 的所有優勢
2. **inpaint** 仍按 9 case + 四錨點機制執行 → 取得策略 B 的訓練可控性
3. 接受極低機率的「inpaint 撞到底圖既有 defect」雜訊（見 [§10.1](#101-撞到機率)）

**Hybrid 不是「折衷」，是「真正能落地的策略 A」**：用 inpaint 彌補策略 A 的四錨點密度問題，同時保留策略 A 的真實底圖優勢。

→ Hybrid 相對策略 B **唯一改變的元件是底圖來源**。其餘（9 case 設計、四錨點機制、模型、loss、訓練流程）完全沿用策略 B。

---

## 5. 訓練 Case 設計

### 5.1 16 case 全分析

訓練 patch 在每個 pixel 上可以是 16 種「前後 pattern 組合」之一。命名規則：`<前pattern>→<後pattern>`，pattern 為 2 位元 `(T, R)`。

> **注意**：下表「inference 遇到？」一欄是基於對製程的**先驗信念**，不是測量結果。Phase 2 早期應該量測真實 case 分布來驗證；若發現某些「罕見」case 其實常見，需要回頭擴充訓練 case set。

| Pattern | 製程情境 | inference 遇到？ | 底圖自然產生？ | GT | Hybrid 處理 |
|---|---|---|---|---|---|
| `00→00` | 正常背景 | 是（大部分像素） | ✅ 是 | 0 | 不必 inpaint |
| `10→10` | 前站 T-only 延續 | 是 | ✅ 真實延續 | 0 | 底圖隱含 + inpaint 補強（**錨點**） |
| `01→01` | 前站 R-only 延續 | 是 | ✅ 真實延續 | 0 | 底圖隱含 + inpaint 補強 |
| `11→11` | 前站 T+R 延續 | 是 | ✅ 真實延續 | 0 | 底圖隱含 + inpaint 補強 |
| `00→10` | **後站新增 T-only（目標！）** | 是（目標） | ❌ 否 | **1** ⭐ | **必須 inpaint** |
| `00→01` | 後站新增 R-only | 是 | ❌ 否 | 0 | 必須 inpaint |
| `00→11` | 後站新增 T+R | 是 | ❌ 否 | 0 | 必須 inpaint |
| `10→00` | 前站 T-only 被製程移除 | 罕見 | △ 可能 | 0 | 不必 inpaint |
| `01→00` | 前站 R-only 被製程移除 | 罕見 | △ 可能 | 0 | 不必 inpaint |
| `11→00` | 前站 T+R 被製程整體移除 | 罕見 | △ 可能 | 0 | 不必 inpaint |
| `10→01` | 前站 T 消失 + 後站 R 新增 | 罕見 | ❌ 否 | 0 | 不必 inpaint |
| `01→10` | **前站 R 消失 + 後站 T 新增** | 罕見 | ❌ 否 | **1** ⭐ | **必須 inpaint**（**錨點**） |
| `10→11` | 前站 T-only + 後站 R 新增 | 極罕見 | ❌ 否 | 0 | 不必 inpaint |
| `01→11` | 前站 R-only + 後站 T 新增 | 極罕見 | ❌ 否 | 0 | 不必 inpaint |
| `11→10` | 前站 T+R + 後站 R 被選擇性移除 | 是 | △ 可能 | 0 | 底圖隱含 + inpaint 補強（**錨點**） |
| `11→01` | 前站 T+R + 後站 T 被選擇性移除 | 是 | △ 可能 | 0 | 底圖隱含 + inpaint 補強 |

GT 欄按 [§3.1 判斷規則](#31-規則本身)推導：`next_T=1 AND next_R=0 AND prev_T=0` 滿足才 = 1。**全 16 case 中只有 `00→10` 跟 `01→10` GT=1**。

### 5.2 從 16 case 篩成 9 case 的邏輯

7 個 case 被排除訓練：

| 被排除的 case | 原因 |
|---|---|
| `00→00` | 訓練 patch 大量 pixel 自然就是這個，不需要主動 inpaint |
| `10→00` / `01→00` / `11→00`（消失類） | inference 「罕見」，先不投入訓練資源 |
| `10→01` | 不算 anomaly（GT=0）且 inference 罕見 |
| `10→11` / `01→11`（疊加類） | inference 極罕見 |

→ **被排除是基於 prior 信念**。Phase 2 早期要驗證這些 case 真的罕見。

剩 9 case 都要 inpaint（即使底圖可能自然產生持續類，inpaint 仍補強以保證每 patch 對等性，理由見 [§5.4](#54-shortcut-風險與防範)）：

| Pattern | GT | 角色 |
|---|---|---|
| `10→10` | 0 | 後站 T-only 家族錨點（前 `10`） |
| `01→01` | 0 | R-only 延續分布覆蓋 |
| `11→11` | 0 | T+R 延續分布覆蓋 |
| `00→10` | **1** ⭐ | 後站 T-only 家族錨點（前 `00`，正樣本 #1） |
| `01→10` | **1** ⭐ | 後站 T-only 家族錨點（前 `01`，正樣本 #2） |
| `00→01` | 0 | 後站新 R-only |
| `00→11` | 0 | 防站內 T 誘惑 |
| `11→10` | 0 | 後站 T-only 家族錨點（前 `11`） |
| `11→01` | 0 | T+R 部分移除分布覆蓋 |

### 5.3 四錨點機制（重點）

**「後站 T-only 家族」**四錨點：`00→10` / `01→10` / `10→10` / `11→10`。

四個 case 在後站站內看**完全一樣**（都是 T-only pattern），唯一差別是前站 pattern：

| Pattern | prev `(T, R)` | GT | 模型該怎麼判斷 |
|---|---|---|---|
| `00→10` | `(0, 0)` | **1** | 真新增（前站完全乾淨） |
| `01→10` | `(0, 1)` | **1** | 真新增（前站只有 R 干擾，T 仍乾淨） |
| `10→10` | `(1, 0)` | 0 | 假新增（前站 T 已有） |
| `11→10` | `(1, 1)` | 0 | 假新增（前站 T+R 都有） |

**唯一能正確區分的訊號是 `prev_T`**：
- `prev_T == 0` → GT=1（前兩個）
- `prev_T == 1` → GT=0（後兩個）

→ 強制每個 patch 都看到這四個 case，逼模型學會「**只看 `prev_T`，忽略 `prev_R`**」這個非對稱判斷。

**dataloader 實作**：每個訓練 patch 至少各放 1 個錨點（`dataloader.py::_assign_cases`）。剩餘的 defect 從 9 case 均勻隨機抽。

> **前提**：此承諾依賴 `num_defects_range` 下限 ≥ 4。dataloader 對下限 < 4 會 print warning 但仍會跑（只放部分 anchor，承諾退化）。`train.sh` / `eval_synthetic.py` 預設 `[4, 10]` 滿足此前提。

### 5.4 Shortcut 風險與防範

若四錨點不齊（例如 patch 內只有 `00→10` 和 `01→10`），模型可能學成：

> **錯誤捷徑**：「找後站 T-only 就標 1」（忽略前站）

這個 shortcut 在「沒有 `10→10` 跟 `11→10` 出現」的情境下無法被 loss 矯正——因為訓練 patch 中所有 `T-only` 都該標 1，模型沒機會學「prev_T=1 時不能標」。

→ 真實 inference 遇到 `10→10`（前站延續類）時系統性誤標。

**Hybrid 的雙重保險**：
1. 底圖中真實前站 defect 自然提供 `10→10` 等持續類的負樣本訓練 signal
2. 即使如此，**仍強制 inpaint 四錨點**，因為：
   - 底圖密度不可控（持續類 0.3 個 events/patch 平均，多數 patch 沒有）
   - **patch 內 case 對等性**比 dataset-wide signal 更重要：模型需要在同一 patch 內看到視覺相似但 GT 不同的對照組，才能學會非對稱判斷
   - `11→10` 在底圖中極稀疏（製程選擇性移除是中等頻率事件，patch 級幾乎遇不到）

### 5.5 合成流程

`dataloader.py::generate_paired_defects` 每次被呼叫：

1. **50% 機率不放 defect** → GT 全 0
   - Phase 1：底圖無 defect，等同「純背景訓練」
   - Phase 2：底圖有真實前站 defect，這 50% 仍提供持續類負樣本 signal
2. **50% 機率放 4-10 個 defect**：
   - 強制塞 4 個錨點（`00→10` / `01→10` / `10→10` / `11→10`）
   - 剩 0-6 個從 9 case 均勻抽（with replacement）
   - 每個 defect 隨機位置、隨機 magnitude、隨機 sign

→ **GT mask 只對 `00→10` 跟 `01→10` 兩個正樣本 case 標 1**。其他 case inpaint 但不標。

---

## 6. 合成監督的本質

### 6.1 這不是 unsupervised

很多人看到「沒有 GT 檔」會以為這是 unsupervised。**不是**。

這是 **supervised learning，只是監督訊號由 dataloader 動態生成而非預先存檔**：

```
focal_loss(model_output, dynamic_GT)   ← 跟一般 supervised 一樣
                              ↑
                  GT 來自我們已知位置的 inpaint，100% 可信
```

對比三種 paradigm：

| | 真實 anomaly 標註 | unsupervised | **合成監督（本專案）** |
|---|---|---|---|
| GT 來源 | 人工標 | 無（pretext task） | inpaint 位置 |
| GT 可信度 | 受標註品質影響 | 不直接 | 100%（我們放的） |
| 成本 | 高（pixel-level 標註） | 低 | 低 |
| 真實 distribution 對齊 | 高 | 中 | 中（Phase 2）/ 低（Phase 1） |

### 6.2 GT 動態合成機制

`data/` 中**沒有** GT mask 檔。GT 由 dataloader 在每次 `__getitem__` 動態合成：

```
__getitem__(idx)
  │
  ├─ 載入該 patch 的前後站 4 通道
  ├─ generate_paired_defects(channels):
  │    ├─ 50% no-defect → 回傳 (channels, GT 全 0)
  │    └─ 50% inpaint:
  │         ├─ N = random(4, 10)
  │         ├─ cases = [4 個錨點] + random_sample(9 cases, N-4)
  │         ├─ for each defect, case in zip(...):
  │         │    apply local inpaint to channels[k] for k flagged by case
  │         │    if case.is_gt: GT_mask |= local_mask
  │         └─ 回傳 (modified channels, GT_mask)
  │
  └─ stack 4 channels, 回傳 dict
```

→ 同一張底圖在不同 epoch 看到的 GT **不同**（inpaint 隨機）。
→ 等效於**無限資料增強**（300 張底圖 × 隨機 inpaint × 多 epoch ≫ 同等規模的固定資料集）。

### 6.3 為什麼動態合成而不是預存

| 預存固定 GT | 動態合成 GT |
|---|---|
| 一次性合成成本高 | 每次 __getitem__ 重新合成（CPU 在 16×16 局部運算，便宜） |
| 每張底圖只看一種 inpaint 配置 | 每 epoch 看到不同配置 → 無限增強 |
| 跑大 epoch 時模型可能背 specific 位置 | 不會背位置，被迫學一般化 |
| 改 case 設計要重生整批資料 | 改 9 case 表立即生效 |

實作成本：`apply_local_defect_to_background` 只動 32² 局部 numpy 區域，每個 patch 4-10 次 → 每 batch ms 級開銷，可忽略。

### 6.4 能驗證 / 不能驗證的

| ✓ 能驗證 | ✗ 不能驗證 |
|---|---|
| Pipeline 邏輯正確（dataloader、model、loss 無 bug） | Model 對**真實**後站新 defect 的偵測能力 |
| Model 能擬合合成 defect distribution | 合成 PSF 與真實 defect 的 domain gap 影響 |
| 9 case 都被訓練到（透過 case coverage 統計） | 真實 noise 下的 robustness（Phase 1） |
| Phase 2 多驗證：能在真實 noise 下處理合成 defect、能正確忽略真實前站 defect | Phase 2 仍不衡量真實 anomaly 偵測效能（GT 仍是合成）|

**核心 caveat**：合成 AUROC = 1.0 **不保證真實效能**。Phase 2 縮小但不消除這個 gap。要量化真實效能，最終仍需小量人工標註驗證集（[§10.5](#105-後續優化路線)）。

---

## 7. 資料 Pipeline

### 7.1 End-to-end 流程圖

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1（當前）：底圖來源                                       │
│                                                              │
│   BRN grayscale tiff  (1, H, W)                              │
│           │                                                  │
│  scripts/build_paired_dataset.py                             │
│   ├─ stack 2 dies + per-die noise → prev (2, H, W)           │
│   └─ apply process shift + stack → next (2, H, W)            │
│           │                                                  │
│           ▼                                                  │
│  data/{train,val,test}/{prev,next}/<name>.tiff (2, H, W)     │
└──────────────────────────────────────────────────────────────┘
                          │
                          │ (Phase 2: 真實前後站影像對直接放入)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Training-time dataloader                                     │
│                                                              │
│  __getitem__(idx) → (prev_path, next_path, patch_position)   │
│           │                                                  │
│  ┌────────┴────────┐                                         │
│  │ Load + crop     │  (cache hit after epoch 1)              │
│  │ patch (128×128) │                                         │
│  └────────┬────────┘                                         │
│           │                                                  │
│  ┌────────┴────────────────────────────────────────────┐     │
│  │ generate_paired_defects(channels):                  │     │
│  │   50% no-defect → GT all 0                          │     │
│  │   50% inpaint:                                      │     │
│  │     N = rand(4, 10)                                 │     │
│  │     cases = anchors(4) + random_sample(9, N-4)      │     │
│  │     for each defect:                                │     │
│  │       sample PSF from pool, random magnitude+sign   │     │
│  │       apply to channels flagged by case             │     │
│  │       if is_gt: GT_mask |= local_mask               │     │
│  └─────────────────┬────────────────────────────────────┘     │
│                    ▼                                         │
│  stack [prev_T, prev_R, next_T, next_R] → (4, 128, 128)      │
│  + GT_mask (1, 128, 128)                                     │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Model + Loss                                                 │
│                                                              │
│  SegmentationNetwork(in_channels=4, out_channels=2)          │
│   ├─ Encoder: 5 conv blocks + SPPF                           │
│   └─ Decoder: 5 upsample + SEBlock + skip-fuse               │
│                    │                                         │
│                    ▼                                         │
│  logits (B, 2, 128, 128) → softmax → channel 1 = anomaly p   │
│                    │                                         │
│                    ▼                                         │
│  FocalLoss(softmax, GT_mask)  α=0.75, cosine γ schedule      │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Inference (test set)                                         │
│                                                              │
│  sliding-window 4-channel input + center-crop stitching      │
│  → full-image heatmap (H, W)                                 │
│                                                              │
│  Optional: eval_synthetic.py 對 test pair 重新合成 GT，       │
│           算 pixel AUROC                                     │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 `data/` 目錄

```
data/
├── train/{prev,next}/   200 對 (2, H, W) CHW float32 tiff
├── val/{prev,next}/      25 對
├── test/{prev,next}/     25 對
└── manifest.json         紀錄合成 / 配對參數以供重現
```

**檔案格式契約**：
- 每張 2 通道灰階，CHW float32 tiff
- 通道 0 = target die，通道 1 = reference die
- `prev/<name>.tiff` ↔ `next/<name>.tiff` 同名配對（同一片晶圓同一位置）
- 前後站之間**像素對齊**（MVP 假設；Phase 2 真實資料若無法對齊需前置配準）
- **`data/` 完全沒有 GT mask 檔**（GT 動態合成）

### 7.3 PSF defect 生成

沿用 `Background_Removal_Net/src_core/generate_psf.py`：

- 環形光瞳 + Zernike 像差 → FFT → Poisson/Gaussian noise → connected-peak 清理
- 支援純量 PSF（type1-3）與 Richards-Wolf 向量 PSF（type4）
- 用 multiprocess pool 預先生成 PSF defect pool（trainer 啟動時一次性，~1000 個 defect/type）
- Runtime 從 pool 隨機抽取，附隨機 magnitude + 隨機 sign

**Vector PSF intensity 校準**：原 `type4_vector.yaml` intensity `[[8, 12]]` 是真實光學 amplitude 量級，在合成底圖上會讓 SNR < 1。`type4_vector_strong.yaml` 改為 `[60, 80]` 是 pixel-perturbation magnitude 量級。Phase 2 切換真實底圖時 intensity 要**重新校準**（真實影像動態範圍可能不同）。

### 7.4 Train / Val / Test 流程差異

| | Train | Val（trainer 內） | Test（eval_synthetic.py） |
|---|---|---|---|
| Dataloader | `PairedDataset` multi-worker | 同左但 `num_workers=0` ⚠️ | 同左但 `no_defect_prob=0` |
| Inpaint 配置 | 每 epoch 隨機 | seed 鎖定（跨 epoch 不變）| seed=12345，可重現 |
| 用途 | 訓練 | 監控收斂 | 最終 AUROC 評估 |

⚠️ **val_loader 必須 `num_workers=0`**：`evaluate_synthetic` 用 `np.random.seed` 鎖 inpaint 配置，但 DataLoader worker 是獨立 process 不繼承主進程 seed。multi-worker 會讓 val 不可重現。詳見 `trainer.py:244-250` 的註解。

---

## 8. Phase 1 vs Phase 2

這個專案分兩階段是因為**真實前後站資料的取得跟整合不是一步到位**。Phase 1 先驗證 pipeline 正確性，Phase 2 才取得 hybrid 設計的核心優勢。

### 8.1 對比表

| 維度 | **Phase 1（已實作）** | **Phase 2（規劃）** |
|---|---|---|
| 底圖來源 | BRN grayscale + 合成 process shift + 合成 noise | 真實前後站影像對（機台篩過「乾淨」後站） |
| 跨站 noise 結構 | 合成 brightness / offset / Gaussian | 真實製程造成 |
| 真實前站 defect | 沒有 | 自然存在於底圖 |
| 持續類 case 來源 | 全靠 inpaint | 底圖自然 + inpaint 雙重來源 |
| GT 純淨度 | 100%（底圖無 defect，inpaint 不會撞到）| 受機台篩選品質影響 |
| Inpaint 撞到底圖機率 | 0%（底圖無 defect）| 5%-22% 含錯標正樣本的 patch（[§10.1](#101-撞到機率)）|
| 真實 inference 分布對齊 | **低（同策略 B）** | **高** |
| 設計優勢取得 | 只取得 2 通道 pipeline + 9 case + 四錨點機制 | 取得 hybrid 全部優勢 |

### 8.2 Phase 1 ≠ 真正的 hybrid

**重要認知**：Phase 1 的訓練 distribution 跟策略 B **本質等價**——兩者都用合成乾淨底圖 + 動態 inpaint，distribution 是同一個。Phase 1 等於把策略 B 的 2 通道設計實作了（策略 B 在文件存在但從未實作為 code），只差「最後一步把底圖換成真實前後站」。

Phase 1 的價值是：**驗證 hybrid 必備元件可以跑通**——
- 2 通道 die-to-die pipeline 沒 bug
- 9 case 字典正確
- 四錨點機制正確強制
- dataloader 動態合成正確
- model 能 fit 這個訓練 distribution（AUROC = 1.0）

但 **Phase 1 不驗證 hybrid 的核心優勢**（真實 noise robustness、真實前站 defect 處理）——這要 Phase 2 才能驗證。

### 8.3 Phase 2 的切換工作

| 工作項 | 內容 |
|---|---|
| 資料準備 | 機台篩選「乾淨」後站影像、跟前站配對成 prev/next 對、各 2 通道 CHW tiff、同名 |
| 排版 | 放入 `data/{train,val,test}/{prev,next}/`。可選擇手動排版或重寫 `scripts/build_paired_dataset.py` |
| Code 改動 | **不需要**改 `src_core/`（資料 contract 不變）|
| 校準 | `defects/*.yaml` 的 `intensity_abs` 重新估算（真實底圖動態範圍可能跟合成不同）|
| 避撞策略評估 | 視撞到機率實機觀察結果決定是否升級避撞（[§10.1](#101-撞到機率)）|
| 真實效能驗證 | 收集小量人工標註集（[§10.5](#105-後續優化路線)）|

→ Phase 2 的 code-side 改動極小，主要工作在資料側。

---

## 9. 模型與訓練

### 9.1 架構：沿用 BRN

```python
SegmentationNetwork(in_channels=4, out_channels=2)
```

- Background_Removal_Net 同款 encoder-decoder + SPPF + SEBlock
- 第一層 conv 接受 4 通道輸入（這是 v1 → Phase 1 的唯一架構改動）
- Output 2 channels → softmax → channel 1 = anomaly probability

完整架構在 `src_core/model.py`。

### 9.2 為什麼不升級架構

- 真實底圖 + 完整 inpaint 機制下，4 通道 concat 讓 encoder 自己學跨站對比已足夠
- Phase 2 切換到真實底圖後，hybrid 訓練 distribution 才比策略 B 更貼近 inference（Phase 1 distribution 跟策略 B 等價）。**先驗證 Phase 2 真實效能再考慮升級**
- 可能的升級方向（[§10.5](#105-後續優化路線)）：siamese twin encoder、cross-attention、deformable conv

### 9.3 Loss 與訓練流程

- **Loss**：FocalLoss + cosine gamma schedule
  - `alpha=0.75`（給少數類正樣本權重）
  - `gamma_start=2.0, gamma_end=2.0`（cosine schedule 但目前頭尾相同 → 等於固定 gamma=2）
- **Optimizer**：Adam, lr=0.001
- **LR schedule**：MultiStepLR，在 80% 跟 90% epoch 各 ×0.2
- **資料切分**：250 張 → 200 train / 25 val / 25 test (8:1:1)
- **訓練量**：20 epoch × 16 batch_size × ~3600 patch/epoch ≈ 10 分鐘 on RTX 5070 Ti

### 9.4 Val / Test 評估

- **Val**（trainer 內每 epoch 跑）：seeded 動態合成 GT → 計算 pixel AUROC，作為跨 epoch 監控收斂訊號
- **Test 視覺化**（`inference.sh`）：跑 sliding-window inference 輸出 heatmap PNG（Phase 1 因為 test 影像也乾淨，heatmap 預期幾乎全黑，是 sanity check）
- **Test 帶 GT 評估**（`eval_synthetic.py`）：對 test pair 重新合成 GT + 計算 AUROC（用 seed=12345 確保跨次運行一致）

---

## 10. 已知限制與後續方向

### 10.1 撞到機率

**Phase 2 開始**，dataloader 隨機 inpaint 可能撞到底圖既有真實 defect → GT 錯標。

**重算後的數字**（區域重疊模型）：
- 真實前站 defect 佔底圖 ~0.01-0.1% pixel
- 單個 32² inpaint 區域內期望覆蓋 defect pixel 數 ≈ 32² × density = 0.1 ~ 1 個
- 撞到機率 per inpaint ≈ 1 - exp(−0.1 ~ −1) ≈ **10% ~ 64%**
- 每 patch 4-10 個 inpaint → 至少撞到 1 個的累積機率 ≈ **33% ~ ~100%**
- 但只有「inpaint 是 GT=1 正樣本」（`00→10` / `01→10`，佔 ~22%）撞到才會錯標
- → 含錯標正樣本的 patch 比例 ~5% ~ 22%

**第一版策略**：完全不做避撞，接受此雜訊比例。實機若發現 systematic 誤判（heatmap 跟底圖既有 defect 位置高度相關），逐步升級避撞方案：

1. **threshold 避撞**：inpaint 前粗略檢查 `prev_T` 該位置強度，超過 threshold 跳過
2. **detector 避撞**：用前置 detector（rule-based 或 strategy B 訓練的初版模型）標出底圖 defect mask，inpaint 嚴格避開
3. **GT 矯正**：inpaint 後重新計算正確 GT（需知道底圖 defect 位置 + 強度）

### 10.2 機台篩選品質

Phase 2 的「乾淨後站」由機台篩選提供。篩選失效會造成「未檢出的真實 `00→10` 留在底圖中」：
- 該位置真實是 anomaly（GT 應為 1）
- 但 dataloader 不會 inpaint 那裡 → GT mask 標 0
- 模型看到「視覺上是 anomaly + 訓練標籤 = 0」→ 學到「這種特徵不該標」
- → **訓練毒藥，recall 受傷**

緩解：
- 多輪篩選（機台 + strategy B 初版模型粗篩雙重把關）
- 人工抽檢估算漏網率
- 訓練後分析 loss 分布找出疑似有未標 anomaly 的 patches

### 10.3 像素對齊假設

MVP 假設前後站像素級對齊。實際若有 sub-pixel 偏移，模型需要同時學「跨站配準」+「找新 defect」兩件事，難度大幅上升。

緩解：
- Phase 2 早期評估真實資料對齊狀況
- 若無法對齊：前置 image registration，或架構升級到 deformable / correlation-based fusion

### 10.4 真實效能未驗證

合成 AUROC 1.0 不等於真實效能。要量化真實效能必須有人工標註的真實 anomaly 驗證集。

### 10.5 後續優化路線

依優先順序：

1. **Phase 2 切換**：取得真實 noise + 真實前站 defect 處理能力
2. **小量人工標註驗證集**：量化真實效能 vs Phase 1 baseline
3. **避撞升級**（若 §10.1 在實機表現出問題）
4. **加入「罕見」case**（若 §5 prior 信念被 Phase 2 量測推翻）
5. **架構升級**（若 4 通道 concat encoder 在真實資料上不夠）

---

## 11. 實作 Status

### 11.1 Phase 1 已完成（commit `b892a97` → `bace3b2`）

| 元件 | 改動 |
|---|---|
| `src_core/model.py` | 改為 4 通道輸入 |
| `src_core/dataloader.py` | 改寫為 9 case 字典 + 四錨點機制 + 動態 inpaint |
| `src_core/trainer.py` | val_loader 強制 `num_workers=0`（確保 seed 真正鎖定 inpaint 配置）；defect 數量下限對齊四錨點 |
| `src_core/inference.py` | 4 通道 sliding-window 推論 + 視覺化 |
| `src_core/eval_synthetic.py` | 4 通道評估；defect 數量預設對齊 `train.sh` |
| `scripts/build_paired_dataset.py` | 生成 2 通道 paired tiff（仍合成底圖；Phase 2 待重寫或丟棄） |
| `src_core/loss.py`、`src_core/defects/*.yaml` | 不變 |

### 11.2 Phase 2 待做

| 元件 | 改動 |
|---|---|
| `scripts/build_paired_dataset.py` | 改寫成「接收真實前後站影像對 + 機台篩選結果，排版到 data/」；或乾脆手動排版資料、丟棄此 script |
| `src_core/defects/*.yaml` 的 `intensity_abs` | 重新校準 |
| 避撞策略 | 視實機觀察結果決定是否升級（[§10.1](#101-撞到機率)） |

### 11.3 程式碼地圖

| 檔案 | 角色 |
|---|---|
| `src_core/model.py` | `SegmentationNetwork(in_channels=4, out_channels=2)` |
| `src_core/dataloader.py` | 9 種 CASES、4 錨點機制、動態 inpaint |
| `src_core/loss.py` | FocalLoss + cosine gamma schedule |
| `src_core/trainer.py` | 訓練流程 + `evaluate_synthetic` |
| `src_core/inference.py` | 4 通道 sliding-window 推論 + 視覺化 |
| `src_core/eval_synthetic.py` | test set 動態合成 + AUROC |
| `src_core/generate_psf.py` | PSF defect 生成 |
| `src_core/defects/*.yaml` | PSF type 設定 |
| `src_core/train.sh` / `inference.sh` | 預設 `type4_vector_strong` / `mvp_vector` checkpoint |
| `scripts/build_paired_dataset.py` | 資料準備（Phase 1 合成；Phase 2 重寫或丟棄） |

---

## 附錄

### A. Archive 指南

`docs/archive/` 中有三份歷史設計筆記，記錄當前 hybrid 路線之前評估過的方案：

| 檔案 | 內容 | 為什麼放棄 |
|---|---|---|
| `design_v1_3ch.md` | 最早的 3 通道版本，commit `b892a97` 對應的設計 | 在 Phase 1 重構為 2 通道 die-to-die pipeline，更貼近實際機台輸出格式 |
| `design_strategyA.md` | 純真實底圖路線評估 | 四錨點密度不可控，工程上不可行（[§4.2](#42-純策略-a-的問題四錨點密度不可控)） |
| `design_strategyB.md` | 純合成底圖路線評估 | 可跑通但 distribution gap 太大，不能驗證真實效能（[§4.3](#43-純策略-b-的問題distribution-gap)） |

這些文件保留是因為它們記錄了**決策歷程**——讀者想理解「為什麼是 hybrid 而不是 X」時的完整論證。日常開發不需要讀。

### B. Glossary

| 術語 | 意義 |
|---|---|
| `T` / target die | die-to-die 比對的目標 die |
| `R` / reference die | die-to-die 比對的相鄰參考 die |
| `prev` / 前站 | 多站點製程中先處理的站 |
| `next` / 後站 | 多站點製程中後處理的站，本專案要找新 defect 的對象 |
| Pattern `(T, R)` | 2-bit 表示「(target 有訊號?, ref 有訊號?)」 |
| Case `<prev>→<next>` | 跨站 pattern 組合，例如 `00→10` 表示前站全乾淨、後站 T-only |
| 四錨點 | `00→10` / `01→10` / `10→10` / `11→10` 四個視覺相似但 GT 不同的 case |
| Hybrid | 真實底圖（取 strategy A 優勢）+ 動態 inpaint（取 strategy B 優勢）的設計策略 |
| Phase 1 | 已實作，distribution 等價於 strategy B（合成底圖 + 4 通道 hybrid pipeline） |
| Phase 2 | 規劃中，真實前後站底圖（hybrid 設計的完整形態） |
| 合成監督 | supervised learning，但 GT 由 dataloader 動態 inpaint 而非預存標註 |
