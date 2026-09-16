import os, heapq, math, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

INPUT_DIR = "../data/input"
OUTPUT_DIR = "../data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRIMAL_NODES = f"{INPUT_DIR}/primal_nodes.csv"
PRIMAL_EDGES = f"{INPUT_DIR}/primal_edges.csv"
DUAL_NODES   = f"{INPUT_DIR}/dual_nodes_enriched.csv"
POI_FEATURES = f"{INPUT_DIR}/poi_features_individual.csv"
NODE_POIS    = f"{INPUT_DIR}/node_pois.csv"
WAY_POIS     = f"{INPUT_DIR}/way_pois.csv"
AADT_FILE    = f"{INPUT_DIR}/aadt_estimates.csv"

SEED = 42
T_REF = 1000.0
RESPECT_ONEWAY = False
DEFAULT_SPEED_KMH = 30.0
SPEED_BY_CLASS = {
    'motorway': 100.0, 'motorway_link': 60.0,
    'trunk': 80.0, 'trunk_link': 50.0,
    'primary': 50.0, 'primary_link': 40.0,
    'secondary': 50.0, 'secondary_link': 40.0,
    'tertiary': 50.0, 'tertiary_link': 40.0,
    'unclassified': 30.0, 'residential': 30.0,
    'living_street': 20.0, 'service': 20.0,
}
POI_CATEGORIES = ['health', 'education', 'commercial', 'public_services', 'transport']

N_OD_PAIRS = 50
MIN_OD_KM = 2.0
BBOX_PAD_KM = 2.0
N_STOP_CANDIDATES = 5
MAX_CANDIDATES_TRIED = 30

random.seed(SEED)
np.random.seed(SEED)


def _sid(v):
    s = str(v)
    return s[:-2] if s.endswith('.0') else s


def approx_m(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) * 111000.0


df_primal_nodes = pd.read_csv(PRIMAL_NODES)
df_primal_nodes = df_primal_nodes.dropna(subset=['node_id', 'lat', 'lon'])
node_positions = {int(r.node_id): (float(r.lat), float(r.lon))
                  for r in df_primal_nodes.itertuples()}
osmnode_to_nodeid = {_sid(r.osmnode_id): int(r.node_id)
                     for r in df_primal_nodes.itertuples()}
print(f"Primal nodes: {len(node_positions):,}")

df_aadt = pd.read_csv(AADT_FILE)
df_aadt['osmway_id'] = df_aadt['osmway_id'].map(_sid)
aadt_map = dict(zip(df_aadt['osmway_id'], df_aadt['aadt'].astype(float)))
measured_ways = set(df_aadt.loc[df_aadt['has_sensor'].astype(bool), 'osmway_id'])
print(f"Road sections with AADT: {len(aadt_map):,} "
      f"({len(measured_ways):,} measured, {len(aadt_map) - len(measured_ways):,} estimated)")

df_dual = pd.read_csv(DUAL_NODES)
df_dual['osmway_id'] = df_dual['osmway_id'].map(_sid)
_speed = pd.to_numeric(df_dual['maxspeed'], errors='coerce')
speed_map = dict(zip(df_dual['osmway_id'], _speed))
road_poi_grouped = {r['osmway_id']: {c: int(r.get(c, 0) or 0) for c in POI_CATEGORIES}
                    for _, r in df_dual.iterrows()}

df_poi_ind = pd.read_csv(POI_FEATURES)
poi_id_col = 'segment_id' if 'segment_id' in df_poi_ind.columns else df_poi_ind.columns[0]
poi_type_columns = [c for c in df_poi_ind.columns if c != poi_id_col]
road_poi_individual = {}
_vals = df_poi_ind[poi_type_columns].to_numpy()
_keys = df_poi_ind[poi_id_col].map(_sid).to_numpy()
for i in range(len(_keys)):
    row = _vals[i]
    nz = np.nonzero(row)[0]
    if len(nz):
        road_poi_individual[_keys[i]] = {poi_type_columns[j]: int(row[j]) for j in nz}

available_types = sorted({k for d in road_poi_individual.values() for k in d})
print(f"Individual POI types present: {len(available_types)}")

_pois = []
for f in (NODE_POIS, WAY_POIS):
    if os.path.exists(f):
        _pois.append(pd.read_csv(f, encoding='utf-8-sig'))
df_pois = pd.concat(_pois, ignore_index=True) if _pois else pd.DataFrame(
    columns=['poi_id', 'lat', 'lon', 'type', 'name'])
print(f"POI locations: {len(df_pois):,}")


def _speed_kmh(osmway_id, highway):
    v = speed_map.get(osmway_id, np.nan)
    if v is not None and np.isfinite(v) and v > 0:
        return float(v)
    return SPEED_BY_CLASS.get(str(highway), DEFAULT_SPEED_KMH)


df_edges = pd.read_csv(PRIMAL_EDGES)

adjacency = {}
edge_data = {}
n_measured = n_estimated = n_no_aadt = 0

for r in df_edges.itertuples():
    drv = str(getattr(r, 'driveable', True))
    if drv in ('False', '0', 'nan', 'None'):
        continue
    src, tgt = int(r.source_id), int(r.target_id)
    if src not in node_positions or tgt not in node_positions:
        continue
    way = _sid(r.osmway_id)
    dist = max(float(r.distance), 1.0)

    if way in aadt_map:
        aadt = max(float(aadt_map[way]), 1.0)
        measured = way in measured_ways
        n_measured += int(measured)
        n_estimated += int(not measured)
    else:
        aadt = 1.0
        measured = False
        n_no_aadt += 1

    freeflow_s = dist / (_speed_kmh(way, getattr(r, 'highway', '')) / 3.6)
    rec = {'distance': dist, 'aadt': aadt, 'measured': measured,
           'freeflow_s': freeflow_s, 'road_name': str(getattr(r, 'road_name', '')),
           'osmway_id': way,
           'poi_individual': road_poi_individual.get(way, {}),
           'poi_grouped': road_poi_grouped.get(way, {c: 0 for c in POI_CATEGORIES})}

    oneway = str(getattr(r, 'oneway', False)) in ('True', '1')
    pairs = [(src, tgt)] if (RESPECT_ONEWAY and oneway) else [(src, tgt), (tgt, src)]
    for s, t in pairs:
        adjacency.setdefault(s, []).append(t)
        edge_data[(s, t)] = rec

print(f"Routing graph: {len(adjacency):,} nodes, {len(edge_data):,} directed edges")
print(f"  edges on measured road sections:  {n_measured:,}")
print(f"  edges on estimated road sections: {n_estimated:,}")
if n_no_aadt:
    print(f"  edges with no AADT at all:        {n_no_aadt:,}")


def edge_cost(e, a_d, a_t):
    return e['distance'] / 100.0 * (a_d + a_t * e['aadt'] / T_REF)


def astar_route(src, tgt, a_d, a_t):
    if src not in adjacency or tgt not in adjacency:
        return None
    if src not in node_positions or tgt not in node_positions:
        return None
    tlat, tlon = node_positions[tgt]

    def h(nd):
        la, lo = node_positions[nd]
        return math.sqrt((la - tlat) ** 2 + (lo - tlon) ** 2) * 111000.0 / 100.0 * a_d

    g = {src: 0.0}
    prev = {src: None}
    pq = [(h(src), 0.0, src)]
    vis = set()
    while pq:
        _, cost, node = heapq.heappop(pq)
        if node == tgt:
            path, cur = [], tgt
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        if node in vis:
            continue
        vis.add(node)
        for nb in adjacency.get(node, []):
            if nb in vis:
                continue
            e = edge_data.get((node, nb))
            if e is None:
                continue
            ng = cost + edge_cost(e, a_d, a_t)
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                prev[nb] = node
                heapq.heappush(pq, (ng + h(nb), ng, nb))
    return None


def path_cost(path, a_d, a_t):
    c = 0.0
    for i in range(len(path) - 1):
        e = edge_data.get((path[i], path[i + 1]))
        if e is not None:
            c += edge_cost(e, a_d, a_t)
    return c


def compute_metrics(path):
    dist = traffic = time_s = 0.0
    meas_m = est_m = 0.0
    pi, pg, roads = {}, {c: 0 for c in POI_CATEGORIES}, set()
    for i in range(len(path) - 1):
        e = edge_data.get((path[i], path[i + 1]))
        if e is None:
            continue
        dist += e['distance']
        traffic += e['distance'] * e['aadt']
        time_s += e['freeflow_s']
        if e['measured']:
            meas_m += e['distance']
        else:
            est_m += e['distance']
        w = e['osmway_id']
        if w and w not in roads:
            roads.add(w)
            for k, v in e['poi_individual'].items():
                pi[k] = pi.get(k, 0) + v
            for c in POI_CATEGORIES:
                pg[c] += e['poi_grouped'].get(c, 0)
    return {'distance_m': dist,
            'traffic_exposure': traffic,
            'avg_traffic': traffic / max(dist, 1.0),
            'time_s': time_s,
            'measured_m': meas_m,
            'estimated_m': est_m,
            'estimated_share': est_m / max(dist, 1.0),
            'hops': len(path),
            'roads_visited': len(roads),
            'poi_individual': pi,
            'poi_grouped': pg,
            'poi_total': sum(pi.values())}


def stop_candidates(src, tgt, poi_type):
    sp, tp = node_positions[src], node_positions[tgt]
    direct = approx_m(sp, tp)
    pad = BBOX_PAD_KM / 111.0
    lo_la, hi_la = min(sp[0], tp[0]) - pad, max(sp[0], tp[0]) + pad
    lo_lo, hi_lo = min(sp[1], tp[1]) - pad, max(sp[1], tp[1]) + pad

    inside, outside = [], []
    for (u, v), e in edge_data.items():
        if e['poi_individual'].get(poi_type, 0) <= 0:
            continue
        if u not in node_positions or v not in node_positions:
            continue
        mid = ((node_positions[u][0] + node_positions[v][0]) / 2.0,
               (node_positions[u][1] + node_positions[v][1]) / 2.0)
        detour = approx_m(sp, mid) + approx_m(mid, tp) - direct
        if lo_la <= mid[0] <= hi_la and lo_lo <= mid[1] <= hi_lo:
            inside.append((detour, u, v))
        else:
            outside.append((detour, u, v))
    cands = inside if inside else outside
    cands.sort(key=lambda z: z[0])
    return cands


def pick_stop(src, tgt, poi_type, a_d, a_t, mode='joint'):
    if src not in node_positions or tgt not in node_positions:
        return None
    cands = stop_candidates(src, tgt, poi_type)
    if not cands:
        return None

    routable = []
    for detour, u, v in cands[:MAX_CANDIDATES_TRIED]:
        p1 = astar_route(src, u, a_d, a_t)
        p2 = astar_route(v, tgt, a_d, a_t)
        if p1 and p2:
            routable.append((detour, u, v, p1, p2))
            if mode == 'distance':
                break
            if len(routable) >= N_STOP_CANDIDATES:
                break
    if not routable:
        return None
    if mode == 'distance':
        detour, u, v, p1, p2 = routable[0]
        return {'u': u, 'v': v, 'detour_m': detour, 'path': p1 + p2}

    best = None
    for detour, u, v, p1, p2 in routable:
        full = p1 + p2
        c = path_cost(full, a_d, a_t)
        e = edge_data.get((u, v))
        if e is not None:
            c += edge_cost(e, a_d, a_t)
        if best is None or c < best[0]:
            best = (c, u, v, detour, full)
    return {'u': best[1], 'v': best[2], 'detour_m': best[3], 'path': best[4]}


def make_od_pairs(n_pairs=N_OD_PAIRS, min_km=MIN_OD_KM, seed=SEED):
    rng = random.Random(seed)
    nodes = [nd for nd in adjacency if nd in node_positions]
    pairs, tries = [], 0
    while len(pairs) < n_pairs and tries < n_pairs * 500:
        tries += 1
        s, t = rng.choice(nodes), rng.choice(nodes)
        if s == t:
            continue
        if approx_m(node_positions[s], node_positions[t]) < min_km * 1000:
            continue
        if astar_route(s, t, 1.0, 0.0) is None:
            continue
        pairs.append((s, t))
    return pairs


print("route_setup ready")
