import math
import folium, webbrowser

df_pois = pd.read_csv = "../data/input"
print(f"POI locations loaded: {len(df_pois)}")

osmnode_to_nodeid = dict(zip(df_primal_nodes['osmnode_id'].astype(str), df_primal_nodes['node_id']))

# =============================================
SOURCE_OSM = 'xxxxx'
TARGET_OSM = 'xxxxx'
POI_TYPE = 'xxxxx'
# =============================================

SOURCE_NODE = osmnode_to_nodeid.get(SOURCE_OSM, None)
TARGET_NODE = osmnode_to_nodeid.get(TARGET_OSM, None)
print(f"Source: {SOURCE_OSM} -> node_id: {SOURCE_NODE}")
print(f"Target: {TARGET_OSM} -> node_id: {TARGET_NODE}")

def approx_m(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * 111000

if SOURCE_NODE is None or TARGET_NODE is None:
    print("ERROR: Source or Target not found!")
else:
    routes = []
    WAYPOINT = None

    if POI_TYPE:
        # --- STEP 1: Find POI edges of requested type ---
        src_pos = node_positions[SOURCE_NODE]
        tgt_pos = node_positions[TARGET_NODE]
        direct = approx_m(src_pos, tgt_pos)

        # Bounding box: 2 km padding in every direction
            pad = 2.0 / 111.0  # ~2 km in degrees
            min_lat = min(src_pos[0], tgt_pos[0]) - pad
            max_lat = max(src_pos[0], tgt_pos[0]) + pad
            min_lon = min(src_pos[1], tgt_pos[1]) - pad
            max_lon = max(src_pos[1], tgt_pos[1]) + pad
            
            poi_candidates = []
            for (u, v), e in edge_data.items():
                if e.get('poi_individual', {}).get(POI_TYPE, 0) > 0:
                    if u not in node_positions or v not in node_positions:
                        continue
                    mid_lat = (node_positions[u][0] + node_positions[v][0]) / 2
                    mid_lon = (node_positions[u][1] + node_positions[v][1]) / 2
                    # Check bounding box
                    if not (min_lat <= mid_lat <= max_lat and min_lon <= mid_lon <= max_lon):
                        continue
                    mid = (mid_lat, mid_lon)
                    detour = approx_m(src_pos, mid) + approx_m(mid, tgt_pos) - direct
                    poi_candidates.append((detour, u, v))
            
            # Fallback: if no POI found in bounding box, search full network
            if not poi_candidates:
                for (u, v), e in edge_data.items():
                    if e.get('poi_individual', {}).get(POI_TYPE, 0) > 0:
                        if u not in node_positions or v not in node_positions:
                            continue
                        mid = ((node_positions[u][0]+node_positions[v][0])/2,
                               (node_positions[u][1]+node_positions[v][1])/2)
                        detour = approx_m(src_pos, mid) + approx_m(mid, tgt_pos) - direct
                        poi_candidates.append((detour, u, v))

        poi_candidates.sort()
        print(f"Found {len(poi_candidates)} edges containing {POI_TYPE}")

        # --- STEP 2: Pick first routable POI edge (smallest detour) ---
        for detour, u, v in poi_candidates[:30]:
            p1, _ = best_route(adjacency, edge_data, node_positions,
                               SOURCE_NODE, u, 1.0, 0.0, 0.0)
            p2, _ = best_route(adjacency, edge_data, node_positions,
                               v, TARGET_NODE, 1.0, 0.0, 0.0)
            if p1 and p2 and (u, v) in edge_data:
                WAYPOINT = (u, v)
                print(f"Selected POI waypoint edge: ({u} -> {v}), detour ~{detour:.0f}m")
                break

        if WAYPOINT is None:
            print(f"No routable {POI_TYPE} found — falling back to no-POI mode")

    # --- STEP 3: Build 3 routes through SAME waypoint (or normal if no POI) ---
    if POI_TYPE and WAYPOINT:
        wp_u, wp_v = WAYPOINT
        configs = [
            ('shortest_poi',     1.0, 0.0, f'Shortest + {POI_TYPE}',     '#0000FF', 6, None),
            ('least_traffic_poi',0.0, 1.0, f'Least Traffic + {POI_TYPE}','#FF0000', 5, '12'),
            ('balanced_poi',     0.5, 0.5, f'Balanced + {POI_TYPE}',     '#00AA00', 5, None),
        ]
        for cname, ad, at, label, color, wt, dash in configs:
            path1, algo1 = best_route(adjacency, edge_data, node_positions,
                                      SOURCE_NODE, wp_u, ad, at, 0.0)
            path2, algo2 = best_route(adjacency, edge_data, node_positions,
                                      wp_v, TARGET_NODE, ad, at, 0.0)
            if path1 and path2:
                # path1 ends at wp_u; path2 starts at wp_v; concatenation
                # crosses the POI edge (wp_u -> wp_v) automatically
                full_path = path1 + path2
                metrics = compute_metrics(full_path, edge_data)
                routes.append({'config': cname, 'label': label, 'color': color,
                               'weight': wt, 'dash': dash, 'path': full_path,
                               'algo': f"{algo1}|{algo2}", **metrics})
    else:
        configs = [
            ('shortest',      1.0, 0.0, 0.0, 'Shortest Distance', '#0000FF', 6, None),
            ('least_traffic', 0.1, 0.9, 0.0, 'Least Traffic',     '#FF0000', 5, '12'),
            ('balanced',      0.5, 0.5, 0.0, 'Balanced',          '#00AA00', 5, None),
        ]
        for cname, ad, at, ap, label, color, wt, dash in configs:
            path, algo = best_route(adjacency, edge_data, node_positions,
                                    SOURCE_NODE, TARGET_NODE, ad, at, ap)
            if path:
                metrics = compute_metrics(path, edge_data)
                routes.append({'config': cname, 'label': label, 'color': color,
                               'weight': wt, 'dash': dash, 'path': path,
                               'algo': algo, **metrics})

    # --- Print summary ---
    pref_str = POI_TYPE if POI_TYPE else 'None'
    print(f"\nSource: {SOURCE_OSM}, Target: {TARGET_OSM}, Preference: {pref_str}")
    pt_label = POI_TYPE if POI_TYPE else "Total"
    print(f"{'Route':<28s} {'Algorithm':<20s} {'Dist(m)':>8s} {'Avg Traffic':>12s} {'POIs':>5s} {pt_label:>10s}")
    print("-" * 90)
    for r in routes:
        pc = r['poi_individual'].get(POI_TYPE, 0) if POI_TYPE else r['poi_total']
        avg_t = r['traffic_exposure'] / max(r['distance_m'], 1)
        print(f"{r['label']:<28s} {r['algo']:<20s} {r['distance_m']:>8.0f} {avg_t:>12.0f} {r['poi_total']:>5d} {pc:>10d}")
