# =============================================================================
# CELL 1: IMPORTS & PATHS
# =============================================================================
 
import os, json, heapq, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import torch
 
STEP1_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step2\data\segment level"
STEP3_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\Step3\data\segment level"
OUTPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step4"
AADT_CSV = r"D:\Manageable\MY Data\roadRouting New\Madrid\Madrid_AADT_clean.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Step 4: GNN-Enhanced Routing (Madrid)")

import pandas as pd

path = r"D:\Manageable\MY Data\roadRouting New\Madrid\step1\data\segment level"

df1 = pd.read_csv(f"{path}/node_pois.csv")
df2 = pd.read_csv(f"{path}/way_pois.csv")
df_pois = pd.concat([df1, df2], ignore_index=True)
df_pois.to_csv(f"{path}/poi_locations.csv", index=False)
print(f"Node POIs: {len(df1)}")
print(f"Way POIs: {len(df2)}")
print(f"Total POIs: {len(df_pois)}")

# =============================================================================
# CELL 2: LOAD DATA
# =============================================================================
 
primal_data = torch.load(f"{STEP1_DIR}/primal_graph.pt", weights_only=False)
pred_aadt_data = torch.load(f"{STEP3_DIR}/predicted_aadt.pt", weights_only=False)
 
pred_aadt_arr = pred_aadt_data['predicted_aadt'].numpy()
osmway_ids = pred_aadt_data['osmway_ids']
 
df_primal_edges = pd.read_csv(f"{STEP1_DIR}/primal_edges.csv")
df_primal_nodes = pd.read_csv(f"{STEP1_DIR}/primal_nodes.csv")
df_dual_nodes = pd.read_csv(f"{STEP1_DIR}/dual_nodes_enriched.csv")
df_poi_individual = pd.read_csv(f"{STEP1_DIR}/poi_features_individual.csv")
 
# Load real AADT from CSV
df_real_aadt = pd.read_csv(AADT_CSV)
df_real_aadt['osmid_str'] = df_real_aadt['osmid'].apply(lambda x: str(int(x)) if pd.notna(x) else None)
real_aadt_map = dict(zip(df_real_aadt['osmid_str'], df_real_aadt['AADT']))
 
# GNN predicted AADT per osmway_id
pred_aadt_map = {}
for i, oid in enumerate(osmway_ids):
    if i < len(pred_aadt_arr):
        pred_aadt_map[str(oid)] = max(float(pred_aadt_arr[i]), 1.0)
 
# Node positions (lat, lon)
pos = primal_data.pos.numpy()
node_positions = {i: (pos[i][1], pos[i][0]) for i in range(len(pos))}
 
print(f"Primal edges: {len(df_primal_edges)}")
print(f"Real AADT roads: {len(real_aadt_map)}")
print(f"GNN predicted AADT roads: {len(pred_aadt_map)}")

# =============================================================================
# CELL 3: BUILD POI LOOKUP & ROUTING GRAPH
# =============================================================================

POI_CATEGORIES = ['health', 'education', 'commercial', 'public_services', 'transport']

# Individual POI per segment (osmway_id)
poi_id_col = 'segment_id' if 'segment_id' in df_poi_individual.columns else 'road_name'
poi_type_columns = [c for c in df_poi_individual.columns if c != poi_id_col]
id_col = 'osmway_id' if 'osmway_id' in df_dual_nodes.columns else 'name'

road_poi_individual = {}
for _, row in df_poi_individual.iterrows():
    sid = str(row[poi_id_col])
    if sid.endswith('.0'):
        sid = sid[:-2]
    poi_dict = {col: int(row[col]) for col in poi_type_columns if int(row[col]) > 0}
    road_poi_individual[sid] = poi_dict

# Grouped POI per segment
road_poi_grouped = {}
for _, row in df_dual_nodes.iterrows():
    sid = str(row[id_col])
    if sid.endswith('.0'):
        sid = sid[:-2]
    road_poi_grouped[sid] = {cat: int(row.get(cat, 0)) for cat in POI_CATEGORIES}

# Available types
available_types = sorted(set(k for pois in road_poi_individual.values() for k in pois.keys()))
print(f"Individual POI types: {len(available_types)}")
for t in available_types:
    total = sum(p.get(t, 0) for p in road_poi_individual.values())
    roads = sum(1 for p in road_poi_individual.values() if p.get(t, 0) > 0)
    print(f"  {t:<25s}: {total:>5d} on {roads:>4d} segments")

# =============================================================================
# CELL 4: BUILD ROUTING GRAPH
# =============================================================================
 
adjacency = {}
edge_data = {}
 
real_used = 0
gnn_used = 0
 
for _, row in df_primal_edges.iterrows():
    src = int(row['source_id'])
    tgt = int(row['target_id'])
    dist = float(row['distance'])
    driveable = row.get('driveable', True)
    road_name = str(row.get('road_name', ''))
    osmway_id = str(row.get('osmway_id', ''))
 
    if driveable == False or driveable == 0 or str(driveable) == 'False':
        continue
 
    # AADT: real if available, else GNN predicted
    raw_aadt = real_aadt_map.get(osmway_id, 0.0)
    gnn_aadt = pred_aadt_map.get(osmway_id, 0.0)
 
    if raw_aadt > 0:
        final_aadt = raw_aadt
        real_used += 1
    else:
        final_aadt = gnn_aadt
        gnn_used += 1
 
    poi_ind = road_poi_individual.get(osmway_id, {})
    poi_grp = road_poi_grouped.get(osmway_id, {c: 0 for c in POI_CATEGORIES})
 
    for s, t in [(src, tgt), (tgt, src)]:
        if s not in adjacency: adjacency[s] = []
        adjacency[s].append(t)
        edge_data[(s, t)] = {
            'distance': max(dist, 1.0),
            'raw_aadt': max(raw_aadt, 1.0),
            'gnn_aadt': max(final_aadt, 1.0),
            'road_name': road_name,
            'osmway_id': osmway_id,
            'poi_individual': poi_ind,
            'poi_grouped': poi_grp
        }
 
driveable_nodes = set(adjacency.keys())
print(f"Routing graph: {len(adjacency)} nodes")
print(f"Edges with real AADT: {real_used}")
print(f"Edges with GNN AADT: {gnn_used}")

# =============================================================================
# CELL 5: DIJKSTRA & A* ALGORITHMS
# =============================================================================

def dijkstra_route(adj, ed, src, tgt, a_d, a_t, a_p, poi_type=None):
    if src not in adj or tgt not in adj: return None
    dist = {src: 0.0}; prev = {src: None}; pq = [(0.0, src)]; vis = set()
    while pq:
        cost, node = heapq.heappop(pq)
        if node == tgt:
            path = []; cur = tgt
            while cur is not None: path.append(cur); cur = prev[cur]
            return path[::-1]
        if node in vis: continue
        vis.add(node)
        for nb in adj.get(node, []):
            if nb in vis: continue
            e = ed.get((node, nb))
            if e is None: continue
            dc = a_d * (e['distance'] / 100.0)
            tc = a_t * (e['gnn_aadt'] / 1000.0)
            pb = a_p * min(e['poi_individual'].get(poi_type, 0), 5) * 0.5 if poi_type and a_p > 0 else 0
            ec = max(dc + tc - pb, 0.01)
            nc = cost + ec
            if nb not in dist or nc < dist[nb]:
                dist[nb] = nc; prev[nb] = node
                heapq.heappush(pq, (nc, nb))
    return None

def astar_route(adj, ed, npos, src, tgt, a_d, a_t, a_p, poi_type=None):
    if src not in adj or tgt not in adj or src not in npos or tgt not in npos: return None
    tlat, tlon = npos[tgt]
    def h(n):
        if n not in npos: return 0
        la, lo = npos[n]
        return math.sqrt((la-tlat)**2 + (lo-tlon)**2) * 111000 * a_d / 100.0
    g = {src: 0.0}; prev = {src: None}; pq = [(h(src), 0.0, src)]; vis = set()
    while pq:
        f, cost, node = heapq.heappop(pq)
        if node == tgt:
            path = []; cur = tgt
            while cur is not None: path.append(cur); cur = prev[cur]
            return path[::-1]
        if node in vis: continue
        vis.add(node)
        for nb in adj.get(node, []):
            if nb in vis: continue
            e = ed.get((node, nb))
            if e is None: continue
            dc = a_d * (e['distance'] / 100.0)
            tc = a_t * (e['gnn_aadt'] / 1000.0)
            pb = a_p * min(e['poi_individual'].get(poi_type, 0), 5) * 0.5 if poi_type and a_p > 0 else 0
            ec = max(dc + tc - pb, 0.01)
            ng = cost + ec
            if nb not in g or ng < g[nb]:
                g[nb] = ng; prev[nb] = node
                heapq.heappush(pq, (ng + h(nb), ng, nb))
    return None

def compute_metrics(path, ed):
    td = 0.0; tt = 0.0; pi = {}; pg = {c: 0 for c in POI_CATEGORIES}; roads = set()
    for i in range(len(path) - 1):
        e = ed.get((path[i], path[i+1]), {})
        td += e.get('distance', 0); tt += e.get('distance', 0) * e.get('gnn_aadt', 0)
        rn = e.get('osmway_id', '')
        if rn and rn not in roads:
            roads.add(rn)
            for k, v in e.get('poi_individual', {}).items(): pi[k] = pi.get(k, 0) + v
            for c in POI_CATEGORIES: pg[c] += e.get('poi_grouped', {}).get(c, 0)
    return {'distance_m': td, 'traffic_exposure': tt, 'hops': len(path),
            'roads_visited': len(roads), 'poi_individual': pi, 'poi_grouped': pg,
            'poi_total': sum(pi.values())}

def best_route(adj, ed, npos, s, t, ad, at, ap, poi_type=None):
    t0 = time.time(); p1 = dijkstra_route(adj, ed, s, t, ad, at, ap, poi_type); t1 = time.time()-t0
    t0 = time.time(); p2 = astar_route(adj, ed, npos, s, t, ad, at, ap, poi_type); t2 = time.time()-t0
    if p1 and p2:
        m1 = compute_metrics(p1, ed); m2 = compute_metrics(p2, ed)
        c1 = ad*m1['distance_m'] + at*m1['traffic_exposure']/1000 - ap*m1['poi_total']
        c2 = ad*m2['distance_m'] + at*m2['traffic_exposure']/1000 - ap*m2['poi_total']
        return (p2, f"A* ({t2:.3f}s)") if c2 <= c1 else (p1, f"Dijkstra ({t1:.3f}s)")
    return (p1, f"Dijkstra ({t1:.3f}s)") if p1 else (p2, f"A* ({t2:.3f}s)") if p2 else (None, None)

print("Routing functions ready")
