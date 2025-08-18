# fir_mtgnn.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Simple MTGNN architecture
# -------------------------
class MTGNNLayer(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [batch, num_nodes, in_dim]
        # adj: [num_nodes, num_nodes]
        out = torch.einsum("bni,io->bno", x, self.theta)
        out = torch.einsum("bno,nm->bmo", out, adj)
        return out + self.bias


class SimpleMTGNN(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim):
        super().__init__()
        self.gnn1 = MTGNNLayer(num_nodes, in_dim, hidden_dim)
        self.gnn2 = MTGNNLayer(num_nodes, hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        h = self.relu(self.gnn1(x, adj))
        out = self.gnn2(h, adj)
        return out.squeeze(-1)  # [batch, num_nodes]


# -------------------------
# Dataset helper
# -------------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, data, p):
        self.data = data
        self.p = p

    def __len__(self):
        return len(self.data) - self.p

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.p]          # [p, num_nodes]
        y = self.data[idx+self.p]              # [num_nodes]
        return torch.FloatTensor(X), torch.FloatTensor(y)


# -------------------------
# Main function
# -------------------------
def fit_mtgnn(train, test, p, q):
    """
    Fits a simplified MTGNN-style GNN model for multivariate one-step-ahead forecasting.

    Parameters:
        train (pd.DataFrame): Training data (columns = series/nodes).
        test (pd.DataFrame): Test data (same columns as train).
        p (int): Number of lags (window size).
        q (int): Forecast horizon (set 1 for one-step ahead).

    Returns:
        predictions (pd.DataFrame): Out-of-sample 1-step-ahead predictions.
        fitted_values (pd.DataFrame): In-sample fitted values.
    """
    assert q == 1, "This simplified MTGNN implementation supports q=1 (one-step ahead) only."

    # Ensure datetime index
    train = train.copy()
    test = test.copy()
    if not isinstance(train.index, pd.DatetimeIndex):
        train.index = pd.date_range(start="2000-01-01", periods=len(train), freq="D")
        test.index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=len(test), freq="D")

    num_nodes = train.shape[1]
    data_all = np.concatenate([train.values, test.values], axis=0)

    # Create a static adjacency matrix (fully connected graph for simplicity)
    adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

    # Prepare training data
    dataset_train = SlidingWindowDataset(train.values, p)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # Model
    model = SimpleMTGNN(num_nodes=num_nodes, in_dim=p, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train
    model.train()
    for epoch in range(5):
        for X, y in loader_train:
            # X: [batch, p, num_nodes] → rearrange to [batch, num_nodes, p]
            X = X.permute(0, 2, 1)
            optimizer.zero_grad()
            pred = model(X, adj)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

    # In-sample fitted values
    fitted_values = []
    model.eval()
    with torch.no_grad():
        for i in range(len(train)):
            if i < p:
                fitted_values.append([np.nan] * num_nodes)
            else:
                X = torch.FloatTensor(train.values[i-p:i]).unsqueeze(0).permute(0, 2, 1)
                pred = model(X, adj).numpy().flatten()
                fitted_values.append(pred.tolist())
    fitted_values_df = pd.DataFrame(fitted_values, index=train.index, columns=train.columns)

    # Out-of-sample one-step-ahead
    predictions = []
    history = train.values.copy()

    with torch.no_grad():
        for t in range(len(test)):
            X = torch.FloatTensor(history[-p:]).unsqueeze(0).permute(0, 2, 1)
            pred = model(X, adj).numpy().flatten()
            predictions.append(pred.tolist())

            # Append actual observation to history
            history = np.vstack([history, test.values[t]])

    predictions_df = pd.DataFrame(predictions, index=test.index, columns=test.columns)

    return predictions_df, fitted_values_df
