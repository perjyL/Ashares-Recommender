# Ashares-Recommender

面向 A 股（沪深300）成分股的量化推荐小项目：用 AkShare 拉历史行情，做技术指标特征，训练模型预测下一交易日上涨概率，并输出 Buy/Hold/Sell。

## 特性
- 拉取沪深300成分股与个股日线行情（前复权）
- 构建均线、MACD、成交量均线、收益率与波动率等特征
- 支持随机森林 / XGBoost / Transformer（单股 / 联合）多种建模方式
- 批量生成推荐结果并输出 CSV
- 提供价格/均线、MACD、成交量可视化

## 项目结构
```bash
Ashares-Recommender/
├─ Stock_Recommender.py          # 主入口脚本，默认批量生成沪深300推荐结果
├─ Example_DataDisplay.py        # 数据与指标展示示例脚本
├─ README.md
├─ requirements.txt
├─ output/
│  └─ hs300_recommendation.csv   # 示例输出文件
└─ src/
   ├─ config.py                  # 全局配置（指数、日期范围、阈值、模型类型等）
   ├─ data_loader.py             # 数据拉取（成分股列表、个股行情、指数行情）
   ├─ feature_engineering.py     # 技术指标与标签构建
   ├─ model.py                   # 模型训练（随机森林、XGBoost、Transformer、联合 Transformer）
   ├─ predictor.py               # 概率到交易信号映射（随机森林使用）
   ├─ recommender.py             # 沪深300批量推荐与排序
   └─ visualization.py           # 绘图工具
```

## 快速开始
1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行推荐脚本

```bash
python Stock_Recommender.py
```

运行后会在 `output/hs300_recommendation.csv` 生成结果，并打印 Top 10。

## 使用方式
- 批量推荐（默认入口）：
  - `Stock_Recommender.py` 调用 `recommender.hs300_recommendation()`
- 单只股票示例：
  - `Stock_Recommender.py` 中的 `main()`
  - 注意：`main()` 使用 `predictor.make_decision`（依赖 `predict_proba`），适用于随机森林 / XGBoost；若 `MODEL_TYPE = "transformer"`，需调整预测逻辑

## 配置
位置：`src/config.py`
- `INDEX_CODE`：指数代码（默认 `000300`）
- `START_DATE` / `END_DATE`：训练与回测时间范围
- `BUY_THRESHOLD` / `SELL_THRESHOLD`：推荐阈值
- `MODEL_TYPE`：`randomforest` / `xgboost` / `transformer`
- `TRANSFORMER_WINDOW` / `TRANSFORMER_EPOCHS`：Transformer 训练参数
- `USE_JOINT_TRANSFORMER`：是否启用联合训练
- `USE_JOINT_FINETUNE`：是否对联合模型做逐股微调
- `JOINT_FINETUNE_EPOCHS` / `JOINT_FINETUNE_LR`：逐股微调参数

## 输出
`output/hs300_recommendation.csv` 字段说明：
- `Rank`：排序名次
- `Code`：股票代码
- `Name`：股票名称
- `Up_Prob`：预测上涨概率
- `Recommendation`：Buy / Hold / Sell

## 依赖
- `requirements.txt`：akshare、pandas、numpy、scikit-learn、xgboost、matplotlib、ta
- 若启用 Transformer：需额外安装 `torch`、`tqdm`

## 注意
- AkShare 需要网络访问；个别股票数据异常会被跳过
- Transformer 对样本长度有要求，样本过短会报错
- 可视化默认使用 Windows 字体 `SimHei`，非 Windows 系统需自行替换

## 算法简述
先用历史行情构造技术指标特征，并用“下一交易日是否上涨”作为标签，转成一个二分类问题。随机森林 / XGBoost 直接在特征上做分类；Transformer 则用滑动窗口把连续 N 天特征作为序列输入，学习时序关系。若开启联合 Transformer，会把沪深300多只股票的数据合并训练一次，然后对单股取最后窗口做预测。最终把上涨概率映射为 Buy/Hold/Sell，并按概率排序输出推荐列表。
