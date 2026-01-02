import akshare as ak
import pandas as pd
import torch

from src.data_loader import get_stock_history
from src.feature_engineering import add_features
from src.model import train_model, train_transformer_joint
from src.config import MODEL_TYPE, USE_JOINT_TRANSFORMER, BUY_THRESHOLD, SELL_THRESHOLD


# ======================================================
# 全局联合 Transformer（只训练一次）
# ======================================================
JOINT_TRANSFORMER_MODEL = None


# ======================================================
# 投资建议规则
# ======================================================
def get_recommendation(prob):
    if prob >= BUY_THRESHOLD:
        return "Buy"
    elif prob >= SELL_THRESHOLD:
        return "Hold"
    else:
        return "Sell"


# ======================================================
# Transformer 专用预测函数
# ======================================================
def transformer_predict(model, X):
    """
    使用最后 window 天数据做 Transformer 预测
    """
    if len(X) < model.window:
        return None

    X_scaled = model.scaler.transform(X)

    seq = torch.tensor(
        X_scaled[-model.window:],
        dtype=torch.float32
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model(seq)

    # 兼容 sigmoid / softmax 两种输出
    if out.ndim == 2 and out.shape[1] == 2:
        prob = torch.softmax(out, dim=1)[0, 1].item()
    else:
        prob = out.squeeze().item()

    return float(prob)


# ======================================================
# 沪深300 推荐主函数
# ======================================================
def hs300_recommendation():
    global JOINT_TRANSFORMER_MODEL

    hs300 = ak.index_stock_cons_csindex(symbol="000300")

    features = [
        "MA5", "MA10", "MA20",
        "DIF", "DEA", "MACD",
        "VOL_MA5", "Volatility"
    ]

    results = []

    # ==================================================
    # 🚀 联合训练 Transformer（只执行一次）
    # ==================================================
    if MODEL_TYPE == "transformer" and USE_JOINT_TRANSFORMER:
        if JOINT_TRANSFORMER_MODEL is None:
            print("🚀 开始联合训练 Transformer（沪深300 横截面 + 时间）...")

            all_dfs = []
            for _, r in hs300.iterrows():
                try:
                    df_i = get_stock_history(r["成分券代码"])
                    df_i = add_features(df_i)
                    if len(df_i) >= 30:
                        all_dfs.append(df_i)
                except Exception:
                    continue

            JOINT_TRANSFORMER_MODEL = train_transformer_joint(
                all_dfs,
                feature_cols=features
            )

            print("✅ 联合 Transformer 训练完成")

    # ==================================================
    # 📊 逐股票预测
    # ==================================================
    for _, row in hs300.iterrows():
        code = row["成分券代码"]
        name = row["成分券名称"]

        try:
            # 1️⃣ 数据加载
            df = get_stock_history(code)
            df = add_features(df)

            X = df[features]
            y = df["Target"]

            if len(X) < 30:
                raise ValueError("样本过短")

            # 2️⃣ 模型训练（非联合 Transformer）
            if MODEL_TYPE != "transformer" or not USE_JOINT_TRANSFORMER:
                model = train_model(X[:-1], y[:-1], MODEL_TYPE)

            # 3️⃣ === 预测 ===
            if MODEL_TYPE == "transformer":
                model_use = (
                    JOINT_TRANSFORMER_MODEL
                    if USE_JOINT_TRANSFORMER
                    else model
                )

                prob = transformer_predict(model_use, X)
                if prob is None:
                    raise ValueError("Transformer 数据不足")

            else:
                prob = model.predict_proba(X.iloc[[-1]])[0, 1]

            # 4️⃣ 投资建议
            rec = get_recommendation(prob)

            results.append({
                "Code": code,
                "Name": name,
                "Up_Prob": round(prob, 4),
                "Recommendation": rec
            })

            print(f"{code} {name} → {rec} ({prob:.2f})")

        except Exception as e:
            # 🔴 现在会打印真实错误，方便你调试
            print(f"{code} {name} 数据异常：{repr(e)}")
            continue

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("Up_Prob", ascending=False)
    df_result.insert(0, "Rank", range(1, len(df_result) + 1))

    return df_result
