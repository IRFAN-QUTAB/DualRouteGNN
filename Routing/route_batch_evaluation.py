# =============================================================================
# CELL 8: BATCH 50 OD PAIRS (5 POI types, waypoint routing)
# =============================================================================

import math

def pick_waypoint(src, tgt, poi_type, max_candidates=30):
    if src not in node_positions or tgt not in node_positions:
        return None
    src_pos = node_positions[src]
    tgt_pos = node_positions[tgt]
    direct = math.sqrt((src_pos[0]-tgt_pos[0])**2 + (src_pos[1]-tgt_pos[1])**2) * 111000

    cands = []
    for (u, v), e in edge_data.items():
        if e.get('poi_individual', {}).get(poi_type, 0) > 0:
            if u not in node_positions or v not in node_positions:
                continue
            mid = ((node_positions[u][0]+node_positions[v][0])/2,
                   (node_positions[u][1]+node_positions[v][1])/2)
            detour = (math.sqrt((src_pos[0]-mid[0])**2 + (src_pos[1]-mid[1])**2)
                    + math.sqrt((mid[0]-tgt_pos[0])**2 + (mid[1]-tgt_pos[1])**2)) * 111000 - direct
            cands.append((detour, u, v))
    cands.sort()

    for detour, u, v in cands[:max_candidates]:
        p1, _ = best_route(adjacency, edge_data, node_positions, src, u, 1.0, 0.0, 0.0)
        p2, _ = best_route(adjacency, edge_data, node_positions, v, tgt, 1.0, 0.0, 0.0)
        if p1 and p2:
            return (u, v, p1, p2)
    return None

# ---------------------------------------------------------------------------
# Configs and POI types
# ---------------------------------------------------------------------------
configs = [
    ('shortest',      1.0, 0.0),
    ('least_traffic', 0.0, 0.1),
    ('balanced',      0.5, 0.5),
]

poi_types = ['clinic', 'pharmacy', 'fuel', 'restaurant', 'school']

# ---------------------------------------------------------------------------
# Run batch for each POI type
# ---------------------------------------------------------------------------
all_results = []
for poi in poi_types:
    batch_results = []
    skipped = 0

    for idx, (src, tgt) in enumerate(od_pairs):
        wp = pick_waypoint(src, tgt, poi)
        if wp is None:
            skipped += 1
            continue
        wp_u, wp_v, _, _ = wp

        row = {'poi_type': poi, 'od_idx': idx}
        ok = True
        for cname, ad, at in configs:
            p1, _ = best_route(adjacency, edge_data, node_positions, src, wp_u, ad, at, 0.0)
            p2, _ = best_route(adjacency, edge_data, node_positions, wp_v, tgt, ad, at, 0.0)
            if not (p1 and p2):
                ok = False
                break
            full_path = p1 + p2
            m = compute_metrics(full_path, edge_data)
            row[f'{cname}_dist'] = m['distance_m']
            row[f'{cname}_traffic'] = m['traffic_exposure']
            row[f'{cname}_avg_traffic'] = m['traffic_exposure'] / max(m['distance_m'], 1)
        if ok:
            batch_results.append(row)

    df_poi = pd.DataFrame(batch_results)
    if len(df_poi) > 0:
        traffic_red = (1 - df_poi['least_traffic_avg_traffic'].mean()
                         / df_poi['shortest_avg_traffic'].mean()) * 100
        dist_inc = (df_poi['least_traffic_dist'].mean()
                         / df_poi['shortest_dist'].mean() - 1) * 100
        bal_traffic_red = (1 - df_poi['balanced_avg_traffic'].mean()
                             / df_poi['shortest_avg_traffic'].mean()) * 100
        bal_dist_inc = (df_poi['balanced_dist'].mean()
                             / df_poi['shortest_dist'].mean() - 1) * 100
        all_results.append({
            'POI': poi,
            'Pairs': len(df_poi),
            'Shortest_dist': f"{df_poi['shortest_dist'].mean():.0f}",
            'Shortest_traffic': f"{df_poi['shortest_avg_traffic'].mean():.0f}",
            'LT_dist': f"{df_poi['least_traffic_dist'].mean():.0f}",
            'LT_traffic': f"{df_poi['least_traffic_avg_traffic'].mean():.0f}",
            'BAL_dist': f"{df_poi['balanced_dist'].mean():.0f}",
            'BAL_traffic': f"{df_poi['balanced_avg_traffic'].mean():.0f}",
            'LT_traffic_red': f"-{traffic_red:.1f}%",
            'LT_dist_inc': f"+{dist_inc:.1f}%",
            'BAL_traffic_red': f"-{bal_traffic_red:.1f}%",
            'BAL_dist_inc': f"+{bal_dist_inc:.1f}%"
        })
        print(f"{poi}: {len(df_poi)} pairs (skipped {skipped}), "
              f"LT traffic -{traffic_red:.1f}%, dist +{dist_inc:.1f}%")
    else:
        print(f"{poi}: no valid pairs (skipped {skipped})")

df_all = pd.DataFrame(all_results)
print("\n" + "="*70)
print("ROUTING RESULTS (5 POI types, 50 OD pairs each)")
print("="*70)
print(df_all.to_string(index=False))

out_path = os.path.join(OUTPUT_DIR, "batch_route_5poi.csv")
df_all.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
