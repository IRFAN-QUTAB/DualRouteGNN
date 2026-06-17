# =============================================================================
# BASELINES: LR, RF, MLP, GCN vs GAT
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
INPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step2\data\segment level"
data = torch.load(f"{INPUT_DIR}\dual_graph.pt", weights_only=False)

X = data.x.numpy()
y = data.y.numpy()
has_aadt = data.train_mask.numpy()

# Same split as GAT (seed=42, 80/20)
sensor_indices = np.where(has_aadt)[0]
np.random.seed(42)
shuffled = np.random.permutation(sensor_indices)
n_train = int(len(sensor_indices) * 0.80)
train_idx = shuffled[:n_train]
test_idx = shuffled[n_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true > 0
    smape = (2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8)).mean() * 100
    print(f"{name:<20s} MAE={mae:>8.1f}  RMSE={rmse:>8.1f}  R²={r2:>7.4f}  SMAPE={smape:>6.2f}%")
    return {'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape}

results = []

# 1. Linear Regression
lr = LinearRegression().fit(X_train, y_train)
results.append(metrics(y_test, lr.predict(X_test), "Linear Regression"))

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
results.append(metrics(y_test, rf.predict(X_test), "Random Forest"))

# 3. MLP
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42).fit(X_train, y_train)
results.append(metrics(y_test, mlp.predict(X_test), "MLP"))

# 4. GCN
class SimpleGCN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x).squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = SimpleGCN(data.num_node_features).to(device)
data_dev = data.to(device)

y_mean = data.y[train_idx].mean()
y_std = data.y[train_idx].std()
y_scaled = (data.y - y_mean) / y_std

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

optimizer = torch.optim.Adam(gcn.parameters(), lr=0.005)
criterion = nn.HuberLoss(delta=5.0)

for epoch in range(1000):
    gcn.train()
    optimizer.zero_grad()
    out = gcn(data_dev.x, data_dev.edge_index)
    loss = criterion(out[train_mask.to(device)], y_scaled.to(device)[train_mask.to(device)])
    loss.backward()
    optimizer.step()

gcn.eval()
with torch.no_grad():
    pred_gcn = (gcn(data_dev.x, data_dev.edge_index) * y_std + y_mean).cpu().numpy()

results.append(metrics(y_test, np.clip(pred_gcn[test_idx], 0, None), "GCN"))
