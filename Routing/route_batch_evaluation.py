from route_setup import *
import pandas as pd
import numpy as np

POI_TYPES = ['clinic', 'pharmacy', 'fuel', 'restaurant', 'school']
OPTIONS = [('shortest', 1.0, 0.0), ('low_traffic', 0.0, 1.0), ('balanced', 0.5, 0.5)]
ALPHA_SWEEP = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SWEEP_POI = 'clinic'
SENS_POI = 'clinic'
SENS_REPEATS = 20
SENS_ALPHA_T = 0.5
ERROR_BAND_FILE = f"{OUTPUT_DIR}/oof_error_by_band_district.csv"

od_pairs = make_od_pairs()
print(f"OD pairs: {len(od_pairs)}")


def route_with_stop(src, tgt, poi_type, a_d, a_t, mode='joint'):
    if poi_type is None:
        p = astar_route(src, tgt, a_d, a_t)
        return None if p is None else {'path': p, 'u': None, 'v': None}
    return pick_stop(src, tgt, poi_type, a_d, a_t, mode=mode)


rows = []
for poi in POI_TYPES:
    skipped = 0
    for idx, (src, tgt) in enumerate(od_pairs):
        rec = {'poi_type': poi, 'od_idx': idx}
        ok = True
        for cname, a_d, a_t in OPTIONS:
            st = route_with_stop(src, tgt, poi, a_d, a_t)
            if st is None:
                ok = False
                break
            m = compute_metrics(st['path'])
            rec[f'{cname}_dist'] = m['distance_m']
            rec[f'{cname}_traffic'] = m['avg_traffic']
            rec[f'{cname}_time'] = m['time_s']
            rec[f'{cname}_est_share'] = m['estimated_share']
        if ok:
            rows.append(rec)
        else:
            skipped += 1
    print(f"{poi}: {sum(1 for r in rows if r['poi_type'] == poi)} pairs "
          f"(skipped {skipped})")

df_pairs = pd.DataFrame(rows)
df_pairs.to_csv(f"{OUTPUT_DIR}/route_pairs.csv", index=False)

summary = []
for poi in POI_TYPES:
    d = df_pairs[df_pairs['poi_type'] == poi]
    if not len(d):
        continue
    row = {'POI': poi, 'Pairs': len(d),
           'Shortest_dist': d['shortest_dist'].mean(),
           'Shortest_traffic': d['shortest_traffic'].mean(),
           'Shortest_time_min': d['shortest_time'].mean() / 60}
    for cname in ('low_traffic', 'balanced'):
        row[f'{cname}_dist_pct'] = (d[f'{cname}_dist'].mean()
                                    / d['shortest_dist'].mean() - 1) * 100
        row[f'{cname}_traffic_pct'] = (d[f'{cname}_traffic'].mean()
                                       / d['shortest_traffic'].mean() - 1) * 100
        row[f'{cname}_time_pct'] = (d[f'{cname}_time'].mean()
                                    / d['shortest_time'].mean() - 1) * 100
    summary.append(row)

df_sum = pd.DataFrame(summary)
df_sum.to_csv(f"{OUTPUT_DIR}/route_summary.csv", index=False)
print("\n" + "=" * 96)
print("ROUTING RESULTS  (change is against the shortest route)")
print("=" * 96)
print(f"{'POI':<12s} {'n':>4s} {'Dist(m)':>9s} {'AADT':>7s} {'Time(min)':>10s} "
      f"| {'LT dist':>8s} {'LT traf':>8s} {'LT time':>8s} "
      f"| {'BAL dist':>9s} {'BAL traf':>9s} {'BAL time':>9s}")
for _, r in df_sum.iterrows():
    print(f"{r['POI']:<12s} {int(r['Pairs']):>4d} {r['Shortest_dist']:>9.0f} "
          f"{r['Shortest_traffic']:>7.0f} {r['Shortest_time_min']:>10.1f} "
          f"| {r['low_traffic_dist_pct']:>+7.1f}% {r['low_traffic_traffic_pct']:>+7.1f}% "
          f"{r['low_traffic_time_pct']:>+7.1f}% "
          f"| {r['balanced_dist_pct']:>+8.1f}% {r['balanced_traffic_pct']:>+8.1f}% "
          f"{r['balanced_time_pct']:>+8.1f}%")

print(f"\nMean share of route length on estimated AADT: "
      f"{df_pairs['shortest_est_share'].mean() * 100:.1f}%")


print("\n" + "=" * 70)
print(f"WEIGHT SWEEP  (POI: {SWEEP_POI})")
print("=" * 70)
sweep = []
for a_t in ALPHA_SWEEP:
    a_d = 1.0 - a_t
    dists, trafs, times = [], [], []
    for src, tgt in od_pairs:
        st = route_with_stop(src, tgt, SWEEP_POI, a_d, a_t)
        if st is None:
            continue
        m = compute_metrics(st['path'])
        dists.append(m['distance_m'])
        trafs.append(m['avg_traffic'])
        times.append(m['time_s'])
    if dists:
        sweep.append({'alpha_t': a_t, 'pairs': len(dists),
                      'distance_m': float(np.mean(dists)),
                      'avg_traffic': float(np.mean(trafs)),
                      'time_s': float(np.mean(times))})

df_sweep = pd.DataFrame(sweep)
if len(df_sweep):
    b = df_sweep.iloc[0]
    df_sweep['dist_pct'] = (df_sweep['distance_m'] / b['distance_m'] - 1) * 100
    df_sweep['traffic_pct'] = (df_sweep['avg_traffic'] / b['avg_traffic'] - 1) * 100
    df_sweep['time_pct'] = (df_sweep['time_s'] / b['time_s'] - 1) * 100
    df_sweep.to_csv(f"{OUTPUT_DIR}/route_weight_sweep.csv", index=False)
    print(f"{'alpha_t':>8s} {'Dist(m)':>9s} {'AADT':>7s} {'Time(min)':>10s} "
          f"{'dDist':>8s} {'dTraffic':>9s} {'dTime':>8s}")
    for _, r in df_sweep.iterrows():
        print(f"{r['alpha_t']:>8.2f} {r['distance_m']:>9.0f} {r['avg_traffic']:>7.0f} "
              f"{r['time_s'] / 60:>10.1f} {r['dist_pct']:>+7.1f}% "
              f"{r['traffic_pct']:>+8.1f}% {r['time_pct']:>+7.1f}%")


print("\n" + "=" * 78)
print("JOINT STOP CHOICE AGAINST CHOOSING THE STOP BY DISTANCE ALONE")
print("=" * 78)
cmp_rows = []
for poi in POI_TYPES:
    for cname, a_d, a_t in OPTIONS:
        same = 0
        dj, dd, tj, td, n_ok = [], [], [], [], 0
        for src, tgt in od_pairs:
            sj = pick_stop(src, tgt, poi, a_d, a_t, mode='joint')
            sd = pick_stop(src, tgt, poi, a_d, a_t, mode='distance')
            if sj is None or sd is None:
                continue
            n_ok += 1
            same += int((sj['u'], sj['v']) == (sd['u'], sd['v']))
            mj, md = compute_metrics(sj['path']), compute_metrics(sd['path'])
            dj.append(mj['distance_m']); dd.append(md['distance_m'])
            tj.append(mj['avg_traffic']); td.append(md['avg_traffic'])
        if n_ok:
            cmp_rows.append({
                'poi_type': poi, 'option': cname, 'pairs': n_ok,
                'same_stop_pct': same / n_ok * 100,
                'dist_change_pct': (np.mean(dj) / np.mean(dd) - 1) * 100,
                'traffic_change_pct': (np.mean(tj) / np.mean(td) - 1) * 100})

df_cmp = pd.DataFrame(cmp_rows)
if len(df_cmp):
    df_cmp.to_csv(f"{OUTPUT_DIR}/route_stop_choice.csv", index=False)
    print(f"{'POI':<12s} {'option':<13s} {'n':>4s} {'same stop':>10s} "
          f"{'dDist':>8s} {'dTraffic':>9s}")
    for _, r in df_cmp.iterrows():
        print(f"{r['poi_type']:<12s} {r['option']:<13s} {int(r['pairs']):>4d} "
              f"{r['same_stop_pct']:>9.1f}% {r['dist_change_pct']:>+7.1f}% "
              f"{r['traffic_change_pct']:>+8.1f}%")
    print("  negative = the joint choice is better")


print("\n" + "=" * 70)
print("SENSITIVITY TO THE ESTIMATION ERROR")
print("=" * 70)
if not os.path.exists(ERROR_BAND_FILE):
    print(f"{ERROR_BAND_FILE} not found - run the GAT file first.")
else:
    bands = pd.read_csv(ERROR_BAND_FILE)
    band_edges = [(float(r['band_low']),
                   np.inf if not np.isfinite(r['band_high']) else float(r['band_high']),
                   float(r['mae'])) for _, r in bands.iterrows()]

    def band_mae(v):
        for lo, hi, mae in band_edges:
            if lo <= v < hi:
                return mae
        return band_edges[-1][2]

    a_d, a_t = 1.0 - SENS_ALPHA_T, SENS_ALPHA_T
    base_routes = {}
    for src, tgt in od_pairs:
        st = pick_stop(src, tgt, SENS_POI, a_d, a_t, mode='joint')
        if st is not None:
            base_routes[(src, tgt)] = (st['path'], compute_metrics(st['path']))
    print(f"Base routes: {len(base_routes)} (alpha_t = {SENS_ALPHA_T})")

    clean = {k: e['aadt'] for k, e in edge_data.items() if not e['measured']}
    rng = np.random.RandomState(SEED)
    d_ch, t_ch, m_ch, sec_ch = [], [], [], []

    for rep in range(SENS_REPEATS):
        by_way = {}
        for (u, v), e in edge_data.items():
            if e['measured']:
                continue
            w = e['osmway_id']
            if w not in by_way:
                base = clean[(u, v)]
                by_way[w] = max(base + rng.normal(0.0, band_mae(base)), 1.0)
        for k, e in edge_data.items():
            if not e['measured']:
                e['aadt'] = by_way.get(e['osmway_id'], e['aadt'])

        for (src, tgt), (bpath, bm) in base_routes.items():
            st = pick_stop(src, tgt, SENS_POI, a_d, a_t, mode='joint')
            if st is None:
                continue
            nm = compute_metrics(st['path'])
            bset, nset = set(bpath), set(st['path'])
            sec_ch.append(1 - len(bset & nset) / max(len(bset | nset), 1))
            d_ch.append(abs(nm['distance_m'] / max(bm['distance_m'], 1) - 1) * 100)
            t_ch.append(abs(nm['avg_traffic'] / max(bm['avg_traffic'], 1e-9) - 1) * 100)
            m_ch.append(abs(nm['time_s'] / max(bm['time_s'], 1e-9) - 1) * 100)

    for k, e in edge_data.items():
        if not e['measured']:
            e['aadt'] = clean[k]

    if d_ch:
        df_sens = pd.DataFrame({'repeats': [SENS_REPEATS],
                                'junctions_changed_pct': [float(np.mean(sec_ch)) * 100],
                                'distance_change_pct': [float(np.mean(d_ch))],
                                'traffic_change_pct': [float(np.mean(t_ch))],
                                'time_change_pct': [float(np.mean(m_ch))]})
        df_sens.to_csv(f"{OUTPUT_DIR}/route_sensitivity.csv", index=False)
        print(f"Junctions on the route that change: {np.mean(sec_ch) * 100:.1f}%")
        print(f"Distance changes by:                {np.mean(d_ch):.1f}%")
        print(f"Traffic changes by:                 {np.mean(t_ch):.1f}%")
        print(f"Time changes by:                    {np.mean(m_ch):.1f}%")

print("\nSaved to " + OUTPUT_DIR)
