# 信用卡欺诈检测 · Credit Card Fraud Detection

一个**开箱即跑**的信用卡欺诈检测项目:合成数据 → 训练 LightGBM → 用 AUC / KS / Precision-Recall 在留出集(holdout)上评估。面向支付风控学习与实践,代码全程中文注释。

> 场景设定:某新加坡电商 / 发卡机构的在线交易反欺诈。方法本身跨国通用;可按 MAS(新加坡金管局)对可疑交易监控的要求扩展规则。

## 为什么用合成数据

业界标准的 ULB 信用卡数据集需要登录 Kaggle 下载。为了让任何人 clone 下来**立即能跑**,本项目自带一个合成数据生成器,模仿 ULB 的结构(`Time, V1..V28, Amount, Class`)和它最关键的特性——**极度类别不平衡(欺诈约 0.3%)**。想换成真实数据见下方"用真实 ULB 数据"。

## 快速开始

```bash
pip install -r requirements.txt
python generate_data.py          # 生成 data/creditcard_synthetic.csv
python train.py                  # 训练 + 评估,产出指标与图
```

产物:
- `reports/metrics.json` —— validation / holdout 的 AUC、KS、PR-AUC,以及所选阈值下的 TP/FP/FN/TN、精确率、召回率
- `reports/figures/roc_ks.png` —— ROC 曲线 + 标注的 KS
- `reports/figures/precision_recall.png` —— PR 曲线(不平衡场景更该看)
- `reports/figures/feature_importance.png` —— 特征重要性

## 流程说明

1. **三段切分** `train 70% / validation 15% / holdout 15%`(分层、固定种子)。**holdout 只在最后用一次**,模拟"未来没见过的交易"。
2. **不平衡处理**:用 `scale_pos_weight` 给稀有的欺诈样本加权。
3. **训练**:LightGBM(梯度提升树),validation 上早停。
4. **选阈值**:在 validation 上用 KS 最大处选切分点(不偷看 holdout)。
5. **评估**(holdout):
   - **AUC** —— 整体排序能力
   - **KS** —— 好坏分布最大区分度 = max(TPR − FPR)
   - **Precision / Recall** —— 在所选阈值下"抓得准不准 / 抓得全不全"
   - **PR-AUC(Average Precision)** —— 极度不平衡时比 ROC-AUC 更能反映实战表现

## 指标怎么读(速查)

| 指标 | 公式 / 含义 | 关注点 |
|---|---|---|
| AUC | 随机一对好/坏,坏的排更前的概率 | 整体排序力,样本不平衡稳健 |
| KS | max(TPR − FPR) | 最佳切分点能把好坏分多开,定审批线 |
| Precision | TP/(TP+FP) | 判为欺诈里真欺诈的比例(误伤反面) |
| Recall(TPR) | TP/(TP+FN) | 真欺诈里被抓到的比例 |
| PR-AUC | PR 曲线下面积 | 低欺诈率下的实战衡量 |

## 用真实 ULB 数据(可选)

1. 从 Kaggle 下载 `Credit Card Fraud Detection`(ULB)数据集的 `creditcard.csv`;
2. 放到 `data/creditcard.csv`;
3. 运行 `python train.py --data data/creditcard.csv`。

字段结构一致,代码无需改动。

## 项目结构

```
generate_data.py   合成数据生成器
train.py           训练 + 评估 + 画图
data/              数据(运行时生成,默认 gitignore,不入库)
reports/           指标与图(运行时生成)
```

## 后续可扩展

- 加入业务规则引擎(夜间大额、异地、短时多笔)做"规则 + 模型"融合
- 用 SHAP 做可解释性,满足模型治理 / 监管解释
- 套用论文方法:序列神经网络审问 → 自动特征发现(见 seq-feature-discovery playbook)
