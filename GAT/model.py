# =============================================================================
# CELL 1: IMPORTS & DEVICE
# =============================================================================

import os, time, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Apple MPS")
else:
    DEVICE = torch.device('cpu')
    print("WARNING: CPU only!")
print(f"Device: {DEVICE}, PyTorch: {torch.__version__}")

# =============================================================================
# CELL 2: CONFIGURATION
# =============================================================================

INPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step2\data\segment level"
OUTPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step3\data\segment level"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 64
NUM_HEADS_1 = 8
NUM_HEADS_2 = 4
NUM_HEADS_3 = 1
DROPOUT = 0.2
EMBEDDING_DIM = 128

EPOCHS = 500
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(RANDOM_SEED)
print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Hidden: {HIDDEN_DIM}, Early stopping: OFF")

# =============================================================================
# CELL 3: MODEL DEFINITION
# =============================================================================

class DualRouteGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, embedding_dim,
                 heads_1, heads_2, heads_3, dropout):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )

        # GAT Layer 1: 8 heads, concat -> 512-d
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads_1,
                            dropout=dropout, concat=True, add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden_dim * heads_1)
        self.skip1 = nn.Linear(hidden_dim, hidden_dim * heads_1)

        # GAT Layer 2: 4 heads, concat -> 256-d
        self.gat2 = GATConv(hidden_dim * heads_1, hidden_dim, heads=heads_2,
                            dropout=dropout, concat=True, add_self_loops=True)
        self.norm2 = nn.LayerNorm(hidden_dim * heads_2)
        self.skip2 = nn.Linear(hidden_dim * heads_1, hidden_dim * heads_2)

        # GAT Layer 3: 1 head, mean -> 128-d
        self.gat3 = GATConv(hidden_dim * heads_2, embedding_dim, heads=heads_3,
                            dropout=dropout, concat=False, add_self_loops=True)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.skip3 = nn.Linear(hidden_dim * heads_2, embedding_dim)

        # Predictor MLP: 128 -> 64 -> 32 -> 1
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, return_attention=False):
        attn = []
        x = self.input_proj(x)

        # GAT Layer 1
        identity = self.skip1(x)
        if return_attention:
            x, (e1, a1) = self.gat1(x, edge_index, return_attention_weights=True)
            attn.append(('gat1', e1, a1))
        else:
            x = self.gat1(x, edge_index)
        x = F.elu(F.dropout(self.norm1(x + identity), p=self.dropout, training=self.training))

        # GAT Layer 2
        identity = self.skip2(x)
        if return_attention:
            x, (e2, a2) = self.gat2(x, edge_index, return_attention_weights=True)
            attn.append(('gat2', e2, a2))
        else:
            x = self.gat2(x, edge_index)
        x = F.elu(F.dropout(self.norm2(x + identity), p=self.dropout, training=self.training))

        # GAT Layer 3
        identity = self.skip3(x)
        if return_attention:
            x, (e3, a3) = self.gat3(x, edge_index, return_attention_weights=True)
            attn.append(('gat3', e3, a3))
        else:
            x = self.gat3(x, edge_index)
        emb = F.elu(self.norm3(x + identity))

        # Predictor
        pred = self.predictor(emb)
        return (pred, emb, attn) if return_attention else (pred, emb)

print("Model defined")

# =============================================================================
# CELL 4: LOAD DATA & PREPARE TRAIN/TEST SPLIT
# =============================================================================

data = torch.load(f"{INPUT_DIR}/dual_graph.pt", weights_only=False)
print(f"Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}, Features: {data.num_node_features}")
print(f"Feature names: {data.feature_names}")

# AADT stats
y_raw = data.y.clone()
has_aadt = data.train_mask.clone()  # train_mask from Step 2 = nodes with real AADT
sensor_indices = torch.where(has_aadt)[0].numpy()
n_sensors = len(sensor_indices)

print(f"\nNodes with real AADT (sensors): {n_sensors:,}")
print(f"Nodes without AADT (predict):  {data.num_nodes - n_sensors:,}")
print(f"AADT range: {y_raw[has_aadt].min():.0f} - {y_raw[has_aadt].max():.0f}")
print(f"AADT mean:  {y_raw[has_aadt].mean():.0f}")

# Z-score normalize target using ONLY sensor nodes
y_sensor = y_raw[has_aadt]
y_mean = y_sensor.mean()
y_std = y_sensor.std()
data.y_scaled = (data.y - y_mean) / y_std
print(f"\nTarget normalization (z-score from sensor nodes only):")
print(f"  y_mean: {y_mean:.2f}, y_std: {y_std:.2f}")

# Split ONLY sensor nodes into train/test (80/20)
np.random.seed(RANDOM_SEED)
shuffled = np.random.permutation(sensor_indices)
n_train = int(n_sensors * TRAIN_RATIO)
train_indices = shuffled[:n_train]
test_indices = shuffled[n_train:]

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask

print(f"\nTrain/Test split (from {n_sensors:,} sensor nodes only):")
print(f"  Train: {train_mask.sum().item():,} ({TRAIN_RATIO*100:.0f}%)")
print(f"  Test:  {test_mask.sum().item():,} ({(1-TRAIN_RATIO)*100:.0f}%)")
print(f"  Unsensored (will predict): {data.num_nodes - n_sensors:,}")

# Move to device
data = data.to(DEVICE)
y_raw_device = y_raw.to(DEVICE)

# Create model (no edge_dim since no edge attributes on dual graph)
model = DualRouteGNN(
    data.num_node_features, HIDDEN_DIM, EMBEDDING_DIM,
    NUM_HEADS_1, NUM_HEADS_2, NUM_HEADS_3, DROPOUT
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.HuberLoss(delta=1.0)

# =============================================================================
# CELL 5: TRAINING
# =============================================================================

training_log = []
best_test_loss = float('inf')
start_time = time.time()
progress = tqdm(range(1, EPOCHS + 1), desc='Training', unit='epoch')

for epoch in progress:
    # Train
    model.train()
    optimizer.zero_grad()
    pred, emb = model(data.x, data.edge_index)
    pred = pred.squeeze()
    loss = criterion(pred[data.train_mask], data.y_scaled[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred, emb = model(data.x, data.edge_index)
        pred = pred.squeeze()
        test_loss = criterion(pred[data.test_mask], data.y_scaled[data.test_mask])

        # Convert back to real scale
        pred_real = pred * y_std.to(DEVICE) + y_mean.to(DEVICE)
        test_mae = F.l1_loss(pred_real[data.test_mask], y_raw_device[data.test_mask]).item()
        train_mae = F.l1_loss(pred_real[data.train_mask], y_raw_device[data.train_mask]).item()
        
        # R2
        ss_res = ((pred_real[data.test_mask] - y_raw_device[data.test_mask]) ** 2).sum()
        ss_tot = ((y_raw_device[data.test_mask] - y_raw_device[data.test_mask].mean()) ** 2).sum()
        test_r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    training_log.append({
        'epoch': epoch,
        'train_loss': loss.item(),
        'test_loss': test_loss.item(),
        'test_mae': test_mae,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'lr': optimizer.param_groups[0]['lr']
    })

    progress.set_postfix({
        'loss': f'{loss.item():.4f}',
        'MAE': f'{test_mae:.1f}',
        'R2': f'{test_r2:.4f}'
    })

    # Save best model
    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_epoch = epoch
        best_mae = test_mae
        best_r2 = test_r2
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_mean': y_mean.item(),
            'y_std': y_std.item()
        }, f"{OUTPUT_DIR}/model_best.pt")

# Save final model
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'y_mean': y_mean.item(),
    'y_std': y_std.item()
}, f"{OUTPUT_DIR}/model_final.pt")

df_log = pd.DataFrame(training_log)
df_log.to_csv(f"{OUTPUT_DIR}/training_log.csv", index=False)
print(f"\nDone in {time.time()-start_time:.1f}s")
print(f"Best epoch: {best_epoch}, MAE: {best_mae:.1f}, R2: {best_r2:.4f}")

# =============================================================================
# CELL 6: TRAINING PLOTS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(df_log['epoch'], df_log['train_loss'], label='Train')
axes[0,0].plot(df_log['epoch'], df_log['test_loss'], label='Test')
axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(df_log['epoch'], df_log['train_mae'], label='Train')
axes[0,1].plot(df_log['epoch'], df_log['test_mae'], label='Test')
axes[0,1].set_title('MAE'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(df_log['epoch'], df_log['test_r2'], color='green')
axes[1,0].axhline(y=0, color='gray', linestyle='--')
axes[1,0].set_title('Test R²'); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(df_log['epoch'], df_log['lr'], color='orange')
axes[1,1].set_yscale('log')
axes[1,1].set_title('Learning Rate'); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('DualRouteGNN Training (Madrid - Name-Based)', fontweight='bold')
plt.tight_layout(); plt.show()

# =============================================================================
# CELL 9: PREDICTED vs ACTUAL PLOT
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Train
tm = data.train_mask.cpu()
axes[0].scatter(y_raw[tm].numpy(), pred_real[tm].numpy(), alpha=0.5, s=20, color='blue')
mx_train = max(y_raw[tm].max(), pred_real[tm].max())
axes[0].plot([0, mx_train], [0, mx_train], 'r--', label='Perfect')
axes[0].set_xlabel('Actual AADT'); axes[0].set_ylabel('Predicted AADT')
axes[0].set_title(f'Train ({tm.sum().item()} nodes)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Test
ts = data.test_mask.cpu()
axes[1].scatter(y_raw[ts].numpy(), pred_real[ts].numpy(), alpha=0.5, s=20, color='green')
mx_test = max(y_raw[ts].max(), pred_real[ts].max())
axes[1].plot([0, mx_test], [0, mx_test], 'r--', label='Perfect')
axes[1].set_xlabel('Actual AADT'); axes[1].set_ylabel('Predicted AADT')
axes[1].set_title(f'Test ({ts.sum().item()} nodes)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('DualRouteGNN: Predicted vs Actual AADT (Madrid)', fontweight='bold')
plt.tight_layout(); plt.show()

print("STEP 3 COMPLETE!")

