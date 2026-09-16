from route_setup import *

SOURCE_OSM = 'xxxxx'
TARGET_OSM = 'xxxxx'
POI_TYPE = 'clinic'
STOP_MODE = 'joint'

OPTIONS = [
    ('shortest',    1.0, 0.0, 'Shortest',    '#0000FF', 6, None),
    ('low_traffic', 0.0, 1.0, 'Low-traffic', '#FF0000', 5, '12'),
    ('balanced',    0.5, 0.5, 'Balanced',    '#00AA00', 5, None),
]

SOURCE_NODE = osmnode_to_nodeid.get(str(SOURCE_OSM))
TARGET_NODE = osmnode_to_nodeid.get(str(TARGET_OSM))
print(f"Source: {SOURCE_OSM} -> node_id {SOURCE_NODE}")
print(f"Target: {TARGET_OSM} -> node_id {TARGET_NODE}")

routes = []
WAYPOINT = None

if SOURCE_NODE is None or TARGET_NODE is None:
    print("Source or target not found in primal_nodes.csv")
else:
    for cname, a_d, a_t, label, color, weight, dash in OPTIONS:
        if POI_TYPE:
            stop = pick_stop(SOURCE_NODE, TARGET_NODE, POI_TYPE, a_d, a_t, mode=STOP_MODE)
            if stop is None:
                print(f"{label}: no routable {POI_TYPE} stop found")
                continue
            path = stop['path']
            if WAYPOINT is None:
                WAYPOINT = (stop['u'], stop['v'])
        else:
            path = astar_route(SOURCE_NODE, TARGET_NODE, a_d, a_t)
            if path is None:
                print(f"{label}: no route found")
                continue
            stop = None

        m = compute_metrics(path)
        routes.append({'config': cname, 'label': label if not POI_TYPE
                       else f"{label} + {POI_TYPE}",
                       'color': color, 'weight': weight, 'dash': dash,
                       'path': path, 'alpha_d': a_d, 'alpha_t': a_t,
                       'stop': stop, **m})

    if routes:
        if WAYPOINT:
            print(f"Stop road section: {WAYPOINT[0]} -> {WAYPOINT[1]} "
                  f"(mode: {STOP_MODE})")
        base = routes[0]
        print(f"\n{'Route':<26s} {'Dist(m)':>9s} {'Avg AADT':>9s} "
              f"{'Time(min)':>10s} {'dDist':>7s} {'dTraffic':>9s} {'dTime':>7s} {'POIs':>5s}")
        print("-" * 92)
        for r in routes:
            dd = (r['distance_m'] / max(base['distance_m'], 1) - 1) * 100
            dt = (r['avg_traffic'] / max(base['avg_traffic'], 1e-9) - 1) * 100
            dm = (r['time_s'] / max(base['time_s'], 1e-9) - 1) * 100
            pc = r['poi_individual'].get(POI_TYPE, 0) if POI_TYPE else r['poi_total']
            print(f"{r['label']:<26s} {r['distance_m']:>9.0f} {r['avg_traffic']:>9.0f} "
                  f"{r['time_s'] / 60:>10.1f} {dd:>+6.1f}% {dt:>+8.1f}% {dm:>+6.1f}% "
                  f"{pc:>5d}")
        print(f"\nShare of the shortest route on estimated AADT: "
              f"{base['estimated_share'] * 100:.1f}%")
