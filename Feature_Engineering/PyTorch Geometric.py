import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
INPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step1\data"    # Where Step 1 saved CSVs
OUTPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step2\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DualRouteGNN - Step 2: CSV to PyTorch Geometric (Madrid)")
print("=" * 60)


# =============================================================
# LOAD CSVs
# =============================================================
print("\n[1] Loading CSV files...")

df_dual_nodes = pd.read_csv(f"{INPUT_DIR}/dual_nodes_enriched.csv")
df_dual_edges = pd.read_csv(f"{INPUT_DIR}/dual_edges.csv")
df_primal_nodes = pd.read_csv(f"{INPUT_DIR}/primal_nodes.csv")
df_primal_edges = pd.read_csv(f"{INPUT_DIR}/primal_edges.csv")

print(f"  Dual nodes:   {len(df_dual_nodes):,}")
print(f"  Dual edges:   {len(df_dual_edges):,}")
print(f"  Primal nodes: {len(df_primal_nodes):,}")
print(f"  Primal edges: {len(df_primal_edges):,}")


# =============================================================
# BUILD DUAL GRAPH (RoadOsm — the main GNN graph)
# =============================================================
print("\n[2] Building DUAL GRAPH (RoadOsm - segment level)...")
print("=" * 60)

# ── Node Features ──
# Numeric features (NO AADT — that is the target, not a feature):
#   distance, pagerank, betweenness, degree
# POI category counts:
#   health, education, commercial, public_services, transport
# Binary:
#   oneway
# Categorical (one-hot):
#   highway type

numeric_cols = ['distance', 'pagerank', 'betweenness', 'degree']
poi_cols = ['health', 'education', 'commercial', 'public_services', 'transport']

print(f"\n  Numeric features: {numeric_cols}")
print(f"  POI features:     {poi_cols}")

# ── Lanes: convert to numeric, fill NaN with median ──
df_dual_nodes['lanes'] = pd.to_numeric(df_dual_nodes['lanes'], errors='coerce')
lanes_nan = df_dual_nodes['lanes'].isna().sum()
lanes_median = df_dual_nodes['lanes'].median()
df_dual_nodes['lanes_filled'] = df_dual_nodes['lanes'].fillna(lanes_median).astype(float)
print(f"\n  Lanes: {lanes_nan} NaN filled with median={lanes_median}")

# ── Maxspeed: convert to numeric, fill NaN with median ──
df_dual_nodes['maxspeed'] = pd.to_numeric(df_dual_nodes['maxspeed'], errors='coerce')
maxspeed_nan = df_dual_nodes['maxspeed'].isna().sum()
maxspeed_median = df_dual_nodes['maxspeed'].median()
df_dual_nodes['maxspeed_filled'] = df_dual_nodes['maxspeed'].fillna(maxspeed_median).astype(float)
print(f"  Maxspeed: {maxspeed_nan} NaN filled with median={maxspeed_median}")

# ── Oneway binary ──
df_dual_nodes['oneway_binary'] = df_dual_nodes['oneway'].apply(
    lambda x: 1.0 if x in [True, 'True', 'true', 'yes'] else 0.0
)
print(f"  Oneway: {df_dual_nodes['oneway_binary'].sum():.0f} one-way out of {len(df_dual_nodes)}")

# ── One-hot encode highway type ──
df_dual_nodes['highway'] = df_dual_nodes['highway'].fillna('unknown')
print(f"\n  Highway types found: {df_dual_nodes['highway'].value_counts().to_dict()}")
road_type_dummies = pd.get_dummies(df_dual_nodes['highway'], prefix='rt')
road_type_cols = list(road_type_dummies.columns)
print(f"  One-hot columns:  {road_type_cols}")

# ── All numeric cols including lanes and maxspeed ──
all_numeric_cols = numeric_cols + ['lanes_filled', 'maxspeed_filled']

# ── Extract and normalize numeric features (min-max) ──
print(f"\n  Normalizing numeric features (min-max)...")
X_numeric = df_dual_nodes[all_numeric_cols].fillna(0).values.astype(np.float32)
for i, col in enumerate(all_numeric_cols):
    col_min = X_numeric[:, i].min()
    col_max = X_numeric[:, i].max()
    if col_max > col_min:
        X_numeric[:, i] = (X_numeric[:, i] - col_min) / (col_max - col_min)
    else:
        X_numeric[:, i] = 0.0
    print(f"    {col}: [{col_min:.2f}, {col_max:.2f}] -> [0, 1]")

# ── Normalize POI features (log1p + min-max) ──
print(f"\n  Normalizing POI features (log1p + min-max)...")
X_poi = df_dual_nodes[poi_cols].fillna(0).values.astype(np.float32)
for i, col in enumerate(poi_cols):
    raw_max = X_poi[:, i].max()
    X_poi[:, i] = np.log1p(X_poi[:, i])
    col_min = X_poi[:, i].min()
    col_max = X_poi[:, i].max()
    if col_max > col_min:
        X_poi[:, i] = (X_poi[:, i] - col_min) / (col_max - col_min)
    else:
        X_poi[:, i] = 0.0
    print(f"    {col}: raw_max={raw_max:.0f}, log1p -> [{col_min:.2f}, {col_max:.2f}] -> [0, 1]")

# ── Oneway ──
X_oneway = df_dual_nodes['oneway_binary'].values.astype(np.float32).reshape(-1, 1)

# ── Highway one-hot ──
X_highway = road_type_dummies.values.astype(np.float32)

# ── Combine ALL features ──
X_dual = np.hstack([X_numeric, X_poi, X_oneway, X_highway])

all_feature_cols = all_numeric_cols + poi_cols + ['oneway'] + road_type_cols

print(f"\n  Feature matrix shape: {X_dual.shape}")
print(f"  = {len(all_numeric_cols)} numeric + {len(poi_cols)} POI + 1 oneway + {len(road_type_cols)} highway one-hot")
print(f"  Total features per node: {X_dual.shape[1]}")

print(f"\n  All features:")
for i, fn in enumerate(all_feature_cols):
    print(f"    x[:, {i}] = {fn}")

x_dual = torch.tensor(X_dual, dtype=torch.float32)

# ── Edge Index ──
source = df_dual_edges['source_id'].values
target = df_dual_edges['target_id'].values
edge_index_dual = torch.tensor(np.array([source, target]), dtype=torch.long)

# ── Target Variable: AADT ──
# Raw AADT for training target
# Nodes WITH sensor data have real AADT > 0
# Nodes WITHOUT sensor data have AADT = 0 (GNN will predict these)
aadt_raw = df_dual_nodes['AADT'].fillna(0).values.astype(np.float32)
y_dual = torch.tensor(aadt_raw, dtype=torch.float32)

# ── Train mask: only nodes with real AADT > 0 ──
has_aadt = aadt_raw > 0
train_mask = torch.tensor(has_aadt, dtype=torch.bool)
print(f"\n  Target (AADT):")
print(f"    Nodes with real AADT (train): {train_mask.sum().item():,}")
print(f"    Nodes without AADT (predict): {(~train_mask).sum().item():,}")
print(f"    AADT range (non-zero): [{aadt_raw[has_aadt].min():.1f}, {aadt_raw[has_aadt].max():.1f}]")
print(f"    AADT mean (non-zero):  {aadt_raw[has_aadt].mean():.1f}")

# ── Create PyG Data Object ──
dual_data = Data(
    x=x_dual,
    edge_index=edge_index_dual,
    y=y_dual,
    num_nodes=len(df_dual_nodes),
    train_mask=train_mask
)

# Store metadata
dual_data.feature_names = all_feature_cols
dual_data.road_names = df_dual_nodes['name'].tolist()
dual_data.osmway_ids = df_dual_nodes['osmway_id'].tolist()

print(f"\n  DUAL GRAPH Data object:")
print(f"    Nodes:         {dual_data.num_nodes:,}")
print(f"    Edges:         {dual_data.num_edges:,}")
print(f"    Node features: {dual_data.num_node_features}")
print(f"    Has target y:  True (AADT)")
print(f"    Has train_mask: True ({train_mask.sum().item():,} train nodes)")

# Connectivity check
nodes_in_edges = set(source.tolist() + target.tolist())
isolated = dual_data.num_nodes - len(nodes_in_edges)
print(f"\n  Connectivity check:")
print(f"    Nodes in edges: {len(nodes_in_edges):,}/{dual_data.num_nodes:,}")
if isolated > 0:
    print(f"    WARNING: {isolated} isolated nodes")


# =============================================================
# BUILD PRIMAL GRAPH (RoadJunction — for routing)
# =============================================================
print("\n\n[3] Building PRIMAL GRAPH (RoadJunction)...")
print("=" * 60)

# ── Node Features ──
primal_numeric_cols = ['poi_count']

print(f"\n  Numeric features: {primal_numeric_cols}")

X_primal = df_primal_nodes[primal_numeric_cols].fillna(0).values.astype(np.float32)

# Add driveable as binary feature
driveable = df_primal_nodes['driveable'].astype(float).values.astype(np.float32)
X_primal = np.hstack([X_primal, driveable.reshape(-1, 1)])
primal_feature_cols = primal_numeric_cols + ['driveable']

print(f"  Feature matrix shape: {X_primal.shape}")

# Normalize numeric features
print(f"\n  Normalizing features...")
for i, col in enumerate(primal_numeric_cols):
    col_min = X_primal[:, i].min()
    col_max = X_primal[:, i].max()
    if col_max > col_min:
        X_primal[:, i] = (X_primal[:, i] - col_min) / (col_max - col_min)
    else:
        X_primal[:, i] = 0.0
    print(f"    {col}: [{col_min:.2f}, {col_max:.2f}] -> [0, 1]")

x_primal = torch.tensor(X_primal, dtype=torch.float32)

# ── Edge Index ──
src_primal = df_primal_edges['source_id'].values
tgt_primal = df_primal_edges['target_id'].values
edge_index_primal = torch.tensor(np.array([src_primal, tgt_primal]), dtype=torch.long)

# ── Edge Attributes (distance, highway type) ──
# Distance (normalized)
dist_vals = df_primal_edges['distance'].fillna(0).values.astype(np.float32)
d_min, d_max = dist_vals.min(), dist_vals.max()
if d_max > d_min:
    dist_norm = (dist_vals - d_min) / (d_max - d_min)
else:
    dist_norm = np.zeros_like(dist_vals)

# Driveable (binary)
drv_edge = df_primal_edges['driveable'].astype(float).values.astype(np.float32)

# Highway type (label encoded then normalized)
le = LabelEncoder()
hw_encoded = le.fit_transform(df_primal_edges['highway'].fillna('unknown'))
hw_norm = hw_encoded.astype(np.float32)
if hw_norm.max() > 0:
    hw_norm = hw_norm / hw_norm.max()

edge_attr_primal = torch.tensor(
    np.stack([dist_norm, drv_edge, hw_norm], axis=1),
    dtype=torch.float32
)

print(f"\n  Edge attributes: distance, driveable, highway_type")
print(f"  Highway classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Spatial coordinates (for visualization, not GNN input) ──
pos_primal = torch.tensor(
    df_primal_nodes[['lon', 'lat']].values.astype(np.float32),
    dtype=torch.float32
)

# ── Create PyG Data Object ──
primal_data = Data(
    x=x_primal,
    edge_index=edge_index_primal,
    edge_attr=edge_attr_primal,
    pos=pos_primal,
    num_nodes=len(df_primal_nodes)
)

primal_data.feature_names = primal_feature_cols
primal_data.edge_feature_names = ['distance', 'driveable', 'highway_type']
primal_data.highway_classes = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
primal_data.osmnode_ids = df_primal_nodes['osmnode_id'].tolist()

print(f"\n  PRIMAL GRAPH Data object:")
print(f"    Nodes:         {primal_data.num_nodes:,}")
print(f"    Edges:         {primal_data.num_edges:,}")
print(f"    Node features: {primal_data.num_node_features}")
print(f"    Edge features: {primal_data.num_edge_features}")
print(f"    Has pos:       True (lat/lon)")


# =============================================================
# SAVE
# =============================================================
print(f"\n\n[4] Saving PyG Data objects...")
print("=" * 60)

torch.save(dual_data, f"{OUTPUT_DIR}/dual_graph.pt")
torch.save(primal_data, f"{OUTPUT_DIR}/primal_graph.pt")

print(f"  Saved -> {OUTPUT_DIR}/dual_graph.pt")
print(f"  Saved -> {OUTPUT_DIR}/primal_graph.pt")

# Save feature info
with open(f"{OUTPUT_DIR}/feature_info.txt", "w") as f:
    f.write("DUAL GRAPH (RoadOsm - Segment Level) FEATURES\n")
    f.write("=" * 50 + "\n")
    f.write("  Each osmway_id segment = one node\n")
    f.write("  Connected if segments share a common RoadJunction\n\n")
    f.write("  Node Features:\n")
    for i, name in enumerate(all_feature_cols):
        f.write(f"    x[:, {i}] = {name}\n")
    f.write(f"\n  y = AADT (raw, unnormalized)\n")
    f.write(f"  train_mask = nodes with real sensor AADT (True = has data)\n")
    f.write(f"  Total nodes: {dual_data.num_nodes:,}\n")
    f.write(f"  Total edges: {dual_data.num_edges:,}\n")
    f.write(f"  Train nodes (with AADT): {train_mask.sum().item():,}\n")
    f.write(f"  Predict nodes (no AADT): {(~train_mask).sum().item():,}\n")

    f.write(f"\n\nPRIMAL GRAPH (RoadJunction) FEATURES\n")
    f.write("=" * 50 + "\n")
    for i, name in enumerate(primal_feature_cols):
        f.write(f"  x[:, {i}] = {name}\n")
    f.write(f"\n  Total nodes: {primal_data.num_nodes:,}\n")
    f.write(f"  Total edges: {primal_data.num_edges:,}\n")
    f.write(f"\n  Edge attributes:\n")
    for i, name in enumerate(['distance', 'driveable', 'highway_type']):
        f.write(f"    edge_attr[:, {i}] = {name}\n")
    f.write(f"\n  Highway type encoding:\n")
    for cls, idx in sorted(primal_data.highway_classes.items(), key=lambda x: x[1]):
        f.write(f"    {idx} = {cls}\n")

print(f"  Saved -> {OUTPUT_DIR}/feature_info.txt")


# =============================================================
# VERIFICATION
# =============================================================
print(f"\n\n[5] Verification...")
print("=" * 60)

d = torch.load(f"{OUTPUT_DIR}/dual_graph.pt", weights_only=False)
p = torch.load(f"{OUTPUT_DIR}/primal_graph.pt", weights_only=False)

print(f"\n  Dual graph reload check:")
print(f"    x shape:          {d.x.shape}")
print(f"    edge_index shape: {d.edge_index.shape}")
print(f"    y shape:          {d.y.shape}")
print(f"    train_mask shape: {d.train_mask.shape}")
print(f"    train_mask True:  {d.train_mask.sum().item():,}")
print(f"    Any NaN in x:     {torch.isnan(d.x).any().item()}")
print(f"    Any NaN in y:     {torch.isnan(d.y).any().item()}")
print(f"    x min/max:        [{d.x.min().item():.4f}, {d.x.max().item():.4f}]")
print(f"    y min/max:        [{d.y.min().item():.1f}, {d.y.max().item():.1f}]")

print(f"\n  Primal graph reload check:")
print(f"    x shape:          {p.x.shape}")
print(f"    edge_index shape: {p.edge_index.shape}")
print(f"    edge_attr shape:  {p.edge_attr.shape}")
print(f"    Any NaN in x:     {torch.isnan(p.x).any().item()}")
print(f"    x min/max:        [{p.x.min().item():.4f}, {p.x.max().item():.4f}]")

# Edge index bounds check
max_dual_idx = d.edge_index.max().item()
max_primal_idx = p.edge_index.max().item()
print(f"\n  Edge index bounds:")
print(f"    Dual:   max={max_dual_idx}, nodes={d.num_nodes}, OK={max_dual_idx < d.num_nodes}")
print(f"    Primal: max={max_primal_idx}, nodes={p.num_nodes}, OK={max_primal_idx < p.num_nodes}")


# =============================================================
# SUMMARY
# =============================================================
print(f"""
{'='*60}
STEP 2 COMPLETE!
{'='*60}

  DUAL GRAPH (for GNN training - segment level):
    Nodes: {dual_data.num_nodes:,} segments
    Edges: {dual_data.num_edges:,} connections
    Features per node: {dual_data.num_node_features}
      - {len(all_numeric_cols)} numeric ({', '.join(all_numeric_cols)})
      - {len(poi_cols)} POI categories ({', '.join(poi_cols)})
      - 1 oneway (binary)
      - {len(road_type_cols)} highway one-hot ({', '.join(road_type_cols)})
    Target: AADT (raw values)
    Train nodes (with sensor data): {train_mask.sum().item():,}
    Predict nodes (no sensor data): {(~train_mask).sum().item():,}

  PRIMAL GRAPH (for routing):
    Nodes: {primal_data.num_nodes:,} junctions
    Edges: {primal_data.num_edges:,} route segments
    Node features: {primal_data.num_node_features} ({', '.join(primal_feature_cols)})
    Edge features: 3 (distance, driveable, highway_type)
    Positions: lat/lon coordinates

  Files in {OUTPUT_DIR}/:
    dual_graph.pt      - Main GNN training data
    primal_graph.pt    - Routing graph data
    feature_info.txt   - Feature reference guide

  Next -> Step 3 (GAT training)
{'='*60}
""")
