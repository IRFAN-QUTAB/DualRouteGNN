import os, math, time, random, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from sklearn.model_selection import KFold, GroupKFold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("GPU:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Apple MPS")
else:
    DEVICE = torch.device('cpu')
    print("CPU only")
print(f"Device: {DEVICE}, PyTorch: {torch.__version__}")


INPUT_DIR = "../data/input"
OUTPUT_DIR = "../data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRAPH_FILE   = f"{INPUT_DIR}/dual_graph.pt"
PRIMAL_NODES = f"{INPUT_DIR}/primal_nodes.csv"
PRIMAL_EDGES = f"{INPUT_DIR}/primal_edges.csv"
DUAL_NODES   = f"{INPUT_DIR}/dual_nodes.csv"

HIDDEN_DIM    = 64
EMBEDDING_DIM = 128
NUM_HEADS_1   = 8
NUM_HEADS_2   = 4
NUM_HEADS_3   = 1
DROPOUT       = 0.2

EPOCHS        = 500
LEARNING_RATE = 0.005
WEIGHT_DECAY  = 1e-4
HUBER_DELTA   = 5.0
SEED          = 42

N_FOLDS       = 5
GRID_KM       = 2.0
N_DISTRICTS   = 5
PROTOCOLS     = ['random', 'spatial_block', 'district', 'corridor']
MAIN_PROTOCOL = 'district'
N_BOOT        = 2000
BOOT_SEED     = 12345


def seed_everything(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything()
print(f"Epochs: {EPOCHS} | folds: {N_FOLDS} | protocols: {PROTOCOLS}")


class DualRouteGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=HIDDEN_DIM, embedding_dim=EMBEDDING_DIM,
                 heads_1=NUM_HEADS_1, heads_2=NUM_HEADS_2, heads_3=NUM_HEADS_3,
                 dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout
        w1, w2, w3 = hidden_dim * heads_1, hidden_dim * heads_2, embedding_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU())

        self.gat1  = GATConv(hidden_dim, w1 // heads_1, heads=heads_1, dropout=dropout,
                             concat=heads_1 > 1, add_self_loops=True)
        self.norm1 = nn.LayerNorm(w1)
        self.skip1 = nn.Linear(hidden_dim, w1)

        self.gat2  = GATConv(w1, w2 // heads_2, heads=heads_2, dropout=dropout,
                             concat=heads_2 > 1, add_self_loops=True)
        self.norm2 = nn.LayerNorm(w2)
        self.skip2 = nn.Linear(w1, w2)

        self.gat3  = GATConv(w2, w3 // heads_3, heads=heads_3, dropout=dropout,
                             concat=heads_3 > 1, add_self_loops=True)
        self.norm3 = nn.LayerNorm(w3)
        self.skip3 = nn.Linear(w2, w3)

        self.predictor = nn.Sequential(
            nn.Linear(w3, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1))

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = F.elu(F.dropout(self.norm1(self.gat1(x, edge_index) + self.skip1(x)),
                            p=self.dropout, training=self.training))
        x = F.elu(F.dropout(self.norm2(self.gat2(x, edge_index) + self.skip2(x)),
                            p=self.dropout, training=self.training))
        emb = F.elu(self.norm3(self.gat3(x, edge_index) + self.skip3(x)))
        return self.predictor(emb).squeeze()


data = torch.load(GRAPH_FILE, weights_only=False)
osmway_ids = [str(w) for w in data.osmway_ids]
road_names = list(data.road_names)
X = data.x.numpy().astype(np.float32)
y = data.y.numpy().astype(np.float32)
sensor = data.train_mask.numpy()
edge_index = data.edge_index
N_NODES = X.shape[0]
print(f"Dual nodes: {N_NODES:,} | sensors: {sensor.sum():,} | features: {X.shape[1]}")


def _ids(col):
    return pd.to_numeric(col, errors='coerce').astype('Int64').astype(str)


pn = pd.read_csv(PRIMAL_NODES, usecols=['osmnode_id', 'lat', 'lon'])
pn = pn.dropna(subset=['osmnode_id', 'lat', 'lon'])
coord = dict(zip(_ids(pn['osmnode_id']), zip(pn['lat'].values, pn['lon'].values)))

pe = pd.read_csv(PRIMAL_EDGES,
                 usecols=['source_osmnode_id', 'target_osmnode_id', 'osmway_id'])
lat_acc, lon_acc = defaultdict(list), defaultdict(list)
for s, t, w in zip(_ids(pe['source_osmnode_id']), _ids(pe['target_osmnode_id']),
                   _ids(pe['osmway_id'])):
    for nd in (s, t):
        if nd in coord:
            la, lo = coord[nd]
            lat_acc[w].append(la)
            lon_acc[w].append(lo)

way_key = list(_ids(pd.Series(osmway_ids)))
lat = np.array([np.mean(lat_acc[w]) if w in lat_acc else np.nan for w in way_key])
lon = np.array([np.mean(lon_acc[w]) if w in lon_acc else np.nan for w in way_key])
print(f"Junctions with coordinates: {len(coord):,} | "
      f"road sections located: {int(np.isfinite(lat).sum()):,}/{len(way_key):,}")
assert np.isfinite(lat).mean() > 0.5, (
    "Most road sections got no coordinate - check that primal_nodes.csv and "
    "primal_edges.csv come from the same city as dual_graph.pt.")

sidx_all = np.where(sensor)[0]
missing = np.isnan(lat[sidx_all]) | np.isnan(lon[sidx_all])
if missing.any():
    print(f"Dropping {int(missing.sum())} sensor road sections without coordinates "
          f"({100 * missing.mean():.1f}% of sensors)")
sidx = sidx_all[~missing]

Xs, ys = X[sidx], y[sidx]
lat_s, lon_s = lat[sidx], lon[sidx]
lat0 = float(np.mean(lat_s))
xkm = (lon_s - np.mean(lon_s)) * 111.0 * math.cos(math.radians(lat0))
ykm = (lat_s - np.mean(lat_s)) * 111.0
XY = np.c_[xkm, ykm]
n = len(sidx)
print(f"Sensors used: {n} | AADT range {ys.min():.0f} - {ys.max():.0f} | "
      f"median {np.median(ys):.0f}")

dn = pd.read_csv(DUAL_NODES, usecols=['osmway_id', 'highway'])
w2c = dict(zip(dn['osmway_id'].astype(str), dn['highway'].astype(str)))
cls_all = np.array([w2c.get(w, 'unknown') for w in osmway_ids])
cls_s = cls_all[sidx]


folds = {}

folds['random'] = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)))

cx = np.floor(xkm / GRID_KM).astype(int)
cy = np.floor(ykm / GRID_KM).astype(int)
_, cells = np.unique(cx * 100000 + cy, return_inverse=True)
ucells = np.unique(cells)
rng = np.random.RandomState(SEED)
rng.shuffle(ucells)
cell2fold = {c: i % N_FOLDS for i, c in enumerate(ucells)}
block_fold = np.array([cell2fold[c] for c in cells])
folds['spatial_block'] = [(np.where(block_fold != k)[0], np.where(block_fold == k)[0])
                          for k in range(N_FOLDS)]

district = KMeans(N_DISTRICTS, random_state=SEED, n_init=10).fit(XY).labels_
folds['district'] = [(np.where(district != k)[0], np.where(district == k)[0])
                     for k in range(N_DISTRICTS)]

grp, name2g, g = [], {}, 0
for i in sidx:
    nm = road_names[i]
    if nm is None or (isinstance(nm, float) and np.isnan(nm)) or str(nm).strip() in ('', 'nan', 'None'):
        grp.append(-1 - int(i))
    else:
        if nm not in name2g:
            name2g[nm] = g
            g += 1
        grp.append(name2g[nm])
grp = np.array(grp)
corridor_fold = np.full(n, -1)
for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(np.arange(n), groups=grp)):
    corridor_fold[te] = k
folds['corridor'] = [(np.where(corridor_fold != k)[0], np.where(corridor_fold == k)[0])
                     for k in range(N_FOLDS)]

for name in PROTOCOLS:
    print(f"{name:14s} test-fold sizes: {[len(te) for _, te in folds[name]]}")


src_np = edge_index[0].numpy()
dst_np = edge_index[1].numpy()


def fold_leakage(tr_local, te_local):
    gtr, gte = sidx[tr_local], sidx[te_local]
    overlap = len(set(gtr.tolist()) & set(gte.tolist()))
    trm = np.zeros(N_NODES, bool); trm[gtr] = True
    tem = np.zeros(N_NODES, bool); tem[gte] = True
    m1 = tem[src_np] & trm[dst_np]
    m2 = tem[dst_np] & trm[src_np]
    if m1.any() or m2.any():
        leaked = np.unique(np.concatenate([src_np[m1], dst_np[m2]]))
    else:
        leaked = np.array([], dtype=int)
    adj_pct = len(leaked) / max(int(tem.sum()), 1) * 100
    d, _ = NearestNeighbors(n_neighbors=1).fit(XY[tr_local]).kneighbors(XY[te_local])
    return overlap, adj_pct, float(np.median(d[:, 0])), float((d[:, 0] < 0.05).mean() * 100)


leak_rows = []
print(f"\n{'protocol':<15s} {'overlap':>8s} {'neighbour-leak%':>16s} "
      f"{'med.nearest-train':>18s} {'test<50m%':>10s}")
print('-' * 72)
for name in PROTOCOLS:
    st = np.array([fold_leakage(tr, te) for tr, te in folds[name]], dtype=float)
    ov, adj, med, near50 = st[:, 0].sum(), st[:, 1].mean(), st[:, 2].mean(), st[:, 3].mean()
    leak_rows.append(dict(split=name, overlap=int(ov), neighbour_leak_pct=adj,
                          median_nearest_train_km=med, test_within_50m_pct=near50))
    print(f"{name:<15s} {int(ov):>8d} {adj:>15.1f}% {med:>15.2f}km {near50:>9.1f}%")

pd.DataFrame(leak_rows).to_csv(f"{OUTPUT_DIR}/leakage_audit.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/leakage_audit.csv")


def metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    y_true = np.asarray(y_true, float)
    return dict(mae=mean_absolute_error(y_true, y_pred),
                rmse=math.sqrt(mean_squared_error(y_true, y_pred)),
                r2=r2_score(y_true, y_pred),
                smape=float((2 * np.abs(y_pred - y_true) /
                             (np.abs(y_pred) + np.abs(y_true) + 1e-8)).mean() * 100))


x_all = torch.tensor(X).to(DEVICE)
ei_all = edge_index.to(DEVICE)
y_t = torch.tensor(y)


def train_predict(train_nodes):
    seed_everything()
    model = DualRouteGNN(X.shape[1]).to(DEVICE)
    ym, ysd = y_t[train_nodes].mean(), y_t[train_nodes].std()
    yz = ((y_t - ym) / ysd).to(DEVICE)
    trm = torch.zeros(N_NODES, dtype=torch.bool)
    trm[train_nodes] = True
    trm_d = trm.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    crit = nn.HuberLoss(delta=HUBER_DELTA)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        p = model(x_all, ei_all)
        crit(p[trm_d], yz[trm_d]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        pred_all = (model(x_all, ei_all).cpu() * ysd + ym).numpy()
    return model, pred_all, float(ym), float(ysd)


print(f"\nModel parameters: "
      f"{sum(p.numel() for p in DualRouteGNN(X.shape[1]).parameters()):,}")


OOF = {}
for split_name in PROTOCOLS:
    t0 = time.time()
    pred = np.full(n, np.nan)
    for tr, te in folds[split_name]:
        _, pred_all, _, _ = train_predict(sidx[tr])
        pred[te] = np.clip(pred_all[sidx[te]], 0, None)
    OOF[split_name] = pred
    ok = ~np.isnan(pred)
    m = metrics(ys[ok], pred[ok])
    print(f"{split_name:<15s} R2={m['r2']:+.3f}  MAE={m['mae']:.0f}  "
          f"RMSE={m['rmse']:.0f}  SMAPE={m['smape']:.1f}%  "
          f"({int(ok.sum())}/{n} roads, {time.time() - t0:.0f}s)")

oof_out = pd.DataFrame({'osmway_id': [osmway_ids[i] for i in sidx],
                        'road_class': cls_s,
                        'aadt_true': ys})
for split_name in PROTOCOLS:
    oof_out[split_name] = OOF[split_name]
oof_out.to_csv(f"{OUTPUT_DIR}/oof_predictions_gat.csv", index=False)
print(f"Saved: {OUTPUT_DIR}/oof_predictions_gat.csv")

rows = []
for split_name in PROTOCOLS:
    ok = ~np.isnan(OOF[split_name])
    rows.append(dict(split=split_name, n_roads=int(ok.sum()),
                     **metrics(ys[ok], OOF[split_name][ok])))
df_pooled = pd.DataFrame(rows)
df_pooled.to_csv(f"{OUTPUT_DIR}/pooled_metrics_gat.csv", index=False)
print(f"\n{'protocol':<15s} {'R2':>8s} {'MAE':>9s} {'RMSE':>9s} {'SMAPE':>8s}")
print('-' * 54)
for _, r in df_pooled.iterrows():
    print(f"{r['split']:<15s} {r['r2']:>+8.3f} {r['mae']:>9.0f} "
          f"{r['rmse']:>9.0f} {r['smape']:>7.1f}%")
spread = df_pooled['r2'].max() - df_pooled['r2'].min()
print(f"\nR2 spread across the four protocols: {spread:.3f}")


rng = np.random.RandomState(BOOT_SEED)
boot_rows = []
for split_name in PROTOCOLS:
    pred = OOF[split_name]
    ok = ~np.isnan(pred)
    yv, pv = ys[ok], pred[ok]
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.randint(0, len(yv), len(yv))
        vals[b] = r2_score(yv[s], pv[s])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    boot_rows.append(dict(split=split_name, r2=r2_score(yv, pv), r2_lo=lo, r2_hi=hi))
    print(f"{split_name:<15s} R2 = {r2_score(yv, pv):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
pd.DataFrame(boot_rows).to_csv(f"{OUTPUT_DIR}/pooled_bootstrap_gat.csv", index=False)


pred = OOF[MAIN_PROTOCOL]
ok = ~np.isnan(pred)
yt, yp = ys[ok], pred[ok]

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.scatter(yt, yp, s=14, alpha=0.45, edgecolor='none')
lim = [0, max(yt.max(), yp.max()) * 1.05]
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Measured AADT (vehicles/day)")
ax.set_ylabel("Estimated AADT (vehicles/day)")
ax.set_title(f"Out-of-fold estimates, {MAIN_PROTOCOL} protocol\n"
             f"n = {int(ok.sum())}, $R^2$ = {r2_score(yt, yp):.3f}, "
             f"MAE = {mean_absolute_error(yt, yp):.0f}")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_scatter_oof_{MAIN_PROTOCOL}.png", dpi=300)
plt.show()

print(f"\nError by traffic level ({MAIN_PROTOCOL} protocol):")
band_rows = []
for lo, hi in [(0, 10000), (10000, 20000), (20000, np.inf)]:
    m = (yt >= lo) & (yt < hi)
    if m.sum() == 0:
        continue
    under = float((yp[m] < yt[m]).mean() * 100)
    band_rows.append(dict(band_low=lo, band_high=hi, roads=int(m.sum()),
                          share_pct=float(m.mean() * 100),
                          mae=float(np.abs(yp[m] - yt[m]).mean()),
                          underestimated_pct=under))
    hi_s = 'inf' if hi == np.inf else f"{hi:,.0f}"
    print(f"  {lo:>7,.0f} - {hi_s:<9s} n={m.sum():>4d} ({m.mean() * 100:4.1f}%)  "
          f"MAE={np.abs(yp[m] - yt[m]).mean():7.0f}  underestimated {under:5.1f}%")
pd.DataFrame(band_rows).to_csv(
    f"{OUTPUT_DIR}/oof_error_by_band_{MAIN_PROTOCOL}.csv", index=False)

print(f"\nError by road class ({MAIN_PROTOCOL} protocol):")
cls_ok = cls_s[ok]
cls_rows = []
for c in pd.Series(cls_ok).value_counts().index:
    m = cls_ok == c
    if m.sum() < 3:
        continue
    cls_rows.append(dict(road_class=c, roads=int(m.sum()),
                         median_aadt=float(np.median(yt[m])),
                         mae=float(np.abs(yp[m] - yt[m]).mean()),
                         r2=float(r2_score(yt[m], yp[m]))))
    print(f"  {c:<16s} n={m.sum():>4d}  median AADT={np.median(yt[m]):>7,.0f}  "
          f"MAE={np.abs(yp[m] - yt[m]).mean():>7,.0f}")
pd.DataFrame(cls_rows).to_csv(
    f"{OUTPUT_DIR}/oof_error_by_class_{MAIN_PROTOCOL}.csv", index=False)


print("\nTraining the final model on every measured road section")
t0 = time.time()
final_model, pred_all, y_mean, y_std = train_predict(sidx)
pred_all = np.clip(pred_all, 0, None)
print(f"Done in {time.time() - t0:.0f}s")

torch.save({'model_state_dict': final_model.state_dict(),
            'in_channels': int(X.shape[1]),
            'hidden_dim': HIDDEN_DIM,
            'embedding_dim': EMBEDDING_DIM,
            'heads': [NUM_HEADS_1, NUM_HEADS_2, NUM_HEADS_3],
            'dropout': DROPOUT,
            'epochs': EPOCHS,
            'y_mean': y_mean,
            'y_std': y_std,
            'feature_names': list(getattr(data, 'feature_names', []))},
           f"{OUTPUT_DIR}/model_final.pt")

has_sensor = np.zeros(N_NODES, bool)
has_sensor[sidx] = True
aadt_final = np.where(has_sensor, y, pred_all)

pd.DataFrame({'osmway_id': osmway_ids,
              'road_class': cls_all,
              'has_sensor': has_sensor,
              'aadt_measured': np.where(has_sensor, y, np.nan),
              'aadt_estimated': pred_all,
              'aadt': aadt_final}).to_csv(
    f"{OUTPUT_DIR}/aadt_estimates.csv", index=False)

print(f"Saved: {OUTPUT_DIR}/model_final.pt")
print(f"Saved: {OUTPUT_DIR}/aadt_estimates.csv")
print(f"Road sections with a measured value: {int(has_sensor.sum()):,}")
print(f"Road sections with an estimate:      {int((~has_sensor).sum()):,}")
