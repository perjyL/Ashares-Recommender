import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TRANSFORMER_WINDOW, TRANSFORMER_EPOCHS

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ========== Random Forest ==========
def train_rf(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


# ========== Transformer ==========
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, window, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.window = window
        self.input_dim = input_dim

        self.embedding = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers
        )
        self.fc = torch.nn.Linear(d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]          # 取最后一天
        x = self.fc(x)
        return self.sigmoid(x).squeeze()


def train_transformer(X, y, window=20, epochs=5):
    """
    Transformer 只训练一次（不逐日）
    """
    if len(X) <= window + 5:
        raise ValueError("样本长度不足以训练 Transformer")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i - window:i])
        y_seq.append(y.iloc[i])

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)

    model = TransformerClassifier(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_seq)
        loss = criterion(output, y_seq)
        loss.backward()
        optimizer.step()

    model.scaler = scaler
    model.window = window
    model.eval()
    return model

# def train_transformer_joint(all_stock_dfs, feature_cols):
#     """
#     all_stock_dfs: List[pd.DataFrame]，每个元素是一只股票的特征DF
#     """
#     X_seq, y_seq = [], []
#
#     for df in all_stock_dfs:
#         if len(df) <= TRANSFORMER_WINDOW:
#             continue
#
#         X = df[feature_cols].values
#         y = df["Target"].values
#
#         for i in range(TRANSFORMER_WINDOW, len(df)):
#             X_seq.append(X[i - TRANSFORMER_WINDOW:i])
#             y_seq.append(y[i])
#
#     X_seq = np.array(X_seq)
#     y_seq = np.array(y_seq)
#
#     scaler = StandardScaler()
#     N, T, F = X_seq.shape
#     X_seq = scaler.fit_transform(X_seq.reshape(-1, F)).reshape(N, T, F)
#
#     X_tensor = torch.tensor(X_seq, dtype=torch.float32)
#     y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
#
#     model = TransformerClassifier(input_dim=F)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.BCELoss()
#
#     model.train()
#     for epoch in range(TRANSFORMER_EPOCHS):
#         optimizer.zero_grad()
#         pred = model(X_tensor)
#         loss = criterion(pred, y_tensor)
#         loss.backward()
#         optimizer.step()
#         print(f"[Transformer Joint] Epoch {epoch+1}, Loss={loss.item():.4f}")
#
#     model.eval()
#     model.scaler = scaler
#     model.window = TRANSFORMER_WINDOW
#     model.feature_cols = feature_cols
#
#     return model

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train_transformer_joint(
    all_dfs,
    feature_cols,
    window=20,
    epochs=8,
    batch_size=64,
    lr=1e-3
):
    print(f"\n📊 联合 Transformer 训练样本构建中...")

    X_all, y_all = [], []

    # ===== 标准化（全市场统一）=====
    N, T, F = X_all.shape
    scaler = StandardScaler()
    X_all_2d = X_all.view(-1, F).numpy()
    X_all_scaled = scaler.fit_transform(X_all_2d)
    X_all = torch.tensor(
        X_all_scaled.reshape(N, T, F),
        dtype=torch.float32
    )

    for df in all_dfs:
        X = df[feature_cols].values
        y = df["Target"].values

        for i in range(window, len(X)):
            X_all.append(X[i - window:i])
            y_all.append(y[i])

    X_all = torch.tensor(np.array(X_all), dtype=torch.float32)
    y_all = torch.tensor(np.array(y_all), dtype=torch.float32)

    print(f"✅ 样本构建完成")
    print(f"   样本数: {len(X_all)}")
    print(f"   Window: {window}")
    print(f"   特征数: {X_all.shape[-1]}")

    dataset = TensorDataset(X_all, y_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerClassifier(
        input_dim=X_all.shape[-1],
        window=window
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    # ===============================
    # 🚀 正式训练（带进度）
    # ===============================
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch}/{epochs}]",
            leave=True
        )

        for X_batch, y_batch in pbar:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch} 完成 | Avg Loss: {avg_loss:.4f}\n")

    print("🎉 联合 Transformer 训练结束\n")

    model.scaler = scaler
    model.window = window
    model.feature_cols = feature_cols
    model.eval()

    return model



# ========== 统一接口 ==========
def train_model(X, y, model_type="randomforest"):
    if model_type == "randomforest":
        return train_rf(X, y)
    elif model_type == "transformer":
        return train_transformer(X, y)
    else:
        raise ValueError("model_type must be 'rf' or 'transformer'")
