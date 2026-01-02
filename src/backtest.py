import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss

from src.data_loader import get_stock_history, get_index_constituents
from src.feature_engineering import add_features
from src.recommender import transformer_predict
from src.model import train_model
from src.config import INDEX_CODE, MODEL_TYPE


FEATURES = [
    "MA5", "MA10", "MA20",
    "MACD", "DIF", "DEA",
    "VOL_MA5", "Volatility"
]


def prob_to_signal(prob):
    if prob >= 0.6:
        return "Buy"
    elif prob >= 0.4:
        return "Hold"
    else:
        return "Sell"


def backtest_single_stock(
    symbol: str,
    start_test_date="2025-01-01",
    end_test_date="2025-12-31",
    min_train_size=200,
    verbose=True
):
    df = get_stock_history(symbol)
    df = add_features(df)

    test_df = df.loc[start_test_date:end_test_date]

    records = []

    if verbose:
        print(f"\n========== 股票 {symbol} ==========")

    for date in test_df.index:
        train_df = df.loc[:date].iloc[:-1]

        if len(train_df) < min_train_size:
            continue

        X_train = train_df[FEATURES]
        y_train = train_df["Target"]

        model = train_model(X_train, y_train, MODEL_TYPE)

        X_test = df.loc[[date], FEATURES]

        if MODEL_TYPE == "transformer":
            prob_up = transformer_predict(model, train_df[FEATURES])
            if prob_up is None:
                continue
        elif MODEL_TYPE == "randomforest":
            prob_up = model.predict_proba(X_test)[0, 1]
        else:
            prob_up = model.predict_proba(X_test)[0, 1]

        pred = int(prob_up >= 0.5)
        signal = prob_to_signal(prob_up)
        true = int(df.loc[date, "Target"])

        correct = "✓" if pred == true else "✗"

        if verbose:
            print(f"[{date.date()}]")
            print(f"训练样本数: {len(train_df)}")
            print(f"上涨概率: {prob_up:.3f}")
            print(f"预测建议: {signal}")
            print(f"真实结果: {'上涨' if true == 1 else '下跌'} {correct}")
            print("-" * 40)

        records.append({
            "date": date,
            "symbol": symbol,
            "prob_up": prob_up,
            "signal": signal,
            "y_true": true,
            "y_pred": pred
        })

    return pd.DataFrame(records)

def backtest_hs300_2025(verbose=True):
    symbols = get_index_constituents(INDEX_CODE)

    print(f"\n开始沪深300回测（股票数量：{len(symbols)}）")
    print("=" * 60)

    all_results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\n>>> [{i}/{len(symbols)}] 回测股票 {symbol}")
        try:
            df = backtest_single_stock(
                symbol,
                verbose=verbose
            )
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            print(f"股票 {symbol} 回测失败：{e}")

    if not all_results:
        return None

    return pd.concat(all_results, ignore_index=True)


def evaluate_overall(result_df: pd.DataFrame):
    acc = accuracy_score(result_df["y_true"], result_df["y_pred"])
    cm = confusion_matrix(result_df["y_true"], result_df["y_pred"])
    brier = brier_score_loss(result_df["y_true"], result_df["prob_up"])

    print("\n========== 沪深300 2025 年整体回测结果 ==========")
    print(f"总体方向预测准确率：{acc:.4f}")
    print("混淆矩阵（真实 x 预测）：")
    print(cm)
    print(f"Brier Score（概率误差）：{brier:.4f}")

    return acc, cm, brier

def backtest_with_equity_curve(df, results):
    """
    df: 含收盘价
    results: 含 signal
    """
    df = df.loc[results.index].copy()
    df["Return"] = df["收盘"].pct_change()

    position_map = {"Buy": 1, "Hold": 0, "Sell": -1}
    df["Position"] = results["signal"].map(position_map)

    df["Strategy_Return"] = df["Position"].shift(1) * df["Return"]
    df.dropna(inplace=True)

    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    df["Market_Equity"] = (1 + df["Return"]).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Strategy_Equity"], label="Strategy")
    plt.plot(df.index, df["Market_Equity"], label="HS300")
    plt.legend()
    plt.title("Cumulative Return Comparison")
    plt.show()
