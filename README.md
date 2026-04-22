# CVPBR — Clustering-based Validity Prediction of Bug Reports
# Reproduction Code 

## 项目结构

```
CVCBR/
├── config.py          # 所有超参数
├── preprocessing.py   # Stage 1: 文本预处理 + Word2Vec（支持 XLSX/CSV）
├── clustering.py      # Stage 2: 谱聚类 + 最优 k（Algorithm 1）
├── adjustment.py      # Stage 3: 聚类调整（Algorithm 2 & 3）
├── model.py           # Stage 4: CNN 模型
├── evaluation.py      # 评估辅助函数
├── main.py            # 完整 Pipeline 入口
├── data/              # ← 放数据（见格式说明）
├── models/            # 自动生成：Word2Vec 权重 + CNN 权重
└── results/           # 自动生成：CSV + JSON 结果
```

---

## 实验协议

### 1. 时间感知划分（Time-aware Split）
- 每个项目的 bug report 按时间列升序排列
- 等分为 **10 段**，前 9 段为训练集，最后 1 段为测试集
- 聚类、调整、模型训练全部在训练集上完成；**测试集对训练过程完全不可见**
- 若数据中无时间列，自动使用行顺序作为时间代理

### 2. 多随机种子（5 Seeds）
- 每个 seed 独立完成：预处理 → Word2Vec → 聚类 → 调整 → CNN 训练 → 测试评估
- 报告 5 个 seed 的**均值 ± 标准差**及 **95% Bootstrap 置信区间**

### 3. 统计检验
- **Wilcoxon 符号秩检验**：检验 CVCBR 与各 baseline 的差异显著性
- **Cliff's δ 效应量**：量化差异大小（negligible/small/medium/large）
- **Bootstrap 95% CI**：关键指标置信区间

---

## 数据格式

每个项目一个 **XLSX 文件**（也支持 CSV），放在 `data/` 目录下：

```
data/
  angular.xlsx
  flutter.xlsx
  ...
```

| 规范列名        | 支持的别名                                      | 类型       | 说明                        |
|-------------|---------------------------------------------|----------|---------------------------|
| `title`     | title, bug_title, summary, subject          | str      | Bug report 标题              |
| `description` | description, desc, body, content, text    | str      | Bug report 描述              |
| `label`     | label, validity, valid, is_valid, status, y | int/str  | 1/valid=有效，0/invalid=无效   |
| 时间列（可选）    | created_at, created, date, timestamp, time  | datetime | 用于时间排序；无则按行顺序            |

---

## 安装依赖

```bash
pip install torch gensim nltk scikit-learn pandas numpy openpyxl scipy
```

---

## 运行

```bash
# 完整 CVCBR（S3 策略，5 seeds，时间感知划分）
python main.py --strategy S3

# 指定数据目录
python main.py --strategy S3 --data_dir ./data

# 消融实验
python main.py --strategy S1   # Algorithm 2 only
python main.py --strategy S2   # Algorithm 2 (alias)

# 自定义 seeds 和时间分段
python main.py --strategy S3 --seeds 42,0,1,2,3 --n_segments 10 --train_segments 9

# 如果时间列列名特殊
python main.py --strategy S3 --time_col "open_date"
```

---
