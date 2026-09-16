from route_compute import *
import os, math
import folium
import webbrowser

MARK_RADIUS_M = 100
NEARBY_PAD_DEG = 0.003
OPEN_IN_BROWSER = True

if not routes:
    print("No routes to draw - run route_compute first.")
else:
    src_pos = node_positions[SOURCE_NODE]
    tgt_pos = node_positions[TARGET_NODE]
    m = folium.Map(location=[(src_pos[0] + tgt_pos[0]) / 2,
                             (src_pos[1] + tgt_pos[1]) / 2], zoom_start=14)

    for r in routes:
        coords = [node_positions[n] for n in r['path'] if n in node_positions]
        if not coords:
            continue
        poi_info = ", ".join(f"{k}={v}" for k, v in r['poi_individual'].items() if v > 0)
        popup = (f"<b>{r['label']}</b><hr>"
                 f"Distance: {r['distance_m']:.0f} m<br>"
                 f"Average AADT: {r['avg_traffic']:.0f} vehicles/day<br>"
                 f"Free-flow time: {r['time_s'] / 60:.1f} min<br>"
                 f"Road sections: {r['roads_visited']}<br>"
                 f"On estimated AADT: {r['estimated_share'] * 100:.0f}%<br>"
                 f"POIs: {r['poi_total']} ({poi_info if poi_info else 'none'})")
        line = folium.PolyLine(coords, color=r['color'], weight=r['weight'],
                               opacity=0.8,
                               popup=folium.Popup(popup, max_width=350),
                               tooltip=f"{r['label']}: {r['distance_m']:.0f} m")
        if r['dash']:
            line.options['dashArray'] = r['dash']
        line.add_to(m)

    on_route = []
    if POI_TYPE and len(df_pois):
        sel = df_pois[df_pois['type'] == POI_TYPE].dropna(subset=['lat', 'lon'])
        route_pts = [node_positions[n] for r in routes for n in r['path']
                     if n in node_positions]
        for _, p in sel.iterrows():
            pl, po = float(p['lat']), float(p['lon'])
            near = any(math.sqrt((pl - a) ** 2 + (po - b) ** 2) * 111000 < MARK_RADIUS_M
                       for a, b in route_pts)
            if near:
                nm = p['name'] if pd.notna(p.get('name')) else POI_TYPE
                folium.Marker([pl, po], popup=f"<b>{POI_TYPE}</b><br>{nm}",
                              icon=folium.Icon(color='purple', icon='plus-sign',
                                               prefix='glyphicon')).add_to(m)
                on_route.append((pl, po))
        print(f"POIs on the route ({POI_TYPE}): {len(on_route)}")

        if route_pts:
            las = [a for a, _ in route_pts]
            los = [b for _, b in route_pts]
            nearby = sel[(sel['lat'] >= min(las) - NEARBY_PAD_DEG) &
                         (sel['lat'] <= max(las) + NEARBY_PAD_DEG) &
                         (sel['lon'] >= min(los) - NEARBY_PAD_DEG) &
                         (sel['lon'] <= max(los) + NEARBY_PAD_DEG)]
            cnt = 0
            for _, p in nearby.iterrows():
                pl, po = float(p['lat']), float(p['lon'])
                if any(abs(a - pl) < 1e-6 and abs(b - po) < 1e-6 for a, b in on_route):
                    continue
                nm = p['name'] if pd.notna(p.get('name')) else POI_TYPE
                folium.CircleMarker([pl, po], radius=3, color='purple', fill=True,
                                    fill_color='purple', fill_opacity=0.7,
                                    popup=f"<b>{POI_TYPE} (nearby)</b><br>{nm}").add_to(m)
                cnt += 1
            print(f"Nearby {POI_TYPE} not on the route: {cnt}")

    folium.Marker(src_pos, popup="<b>ORIGIN</b>",
                  icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
    folium.Marker(tgt_pos, popup="<b>DESTINATION</b>",
                  icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)

    if WAYPOINT and WAYPOINT[0] in node_positions and WAYPOINT[1] in node_positions:
        folium.PolyLine([node_positions[WAYPOINT[0]], node_positions[WAYPOINT[1]]],
                        color='#FF00FF', weight=10, opacity=0.6,
                        tooltip=f"Stop road section ({POI_TYPE})").add_to(m)

    items = ""
    for r in routes:
        items += (f'<span style="color:{r["color"]}; font-size:16px;">&#9644;&#9644;</span>'
                  f'&nbsp; <b>{r["label"]}</b> &mdash; {r["distance_m"]:.0f} m, '
                  f'{r["avg_traffic"]:.0f} veh/day, {r["time_s"] / 60:.1f} min<br>')
    if POI_TYPE:
        items += (f'<br><span style="color:purple;">&#10010;</span> {POI_TYPE} (on route)'
                  f'<br><span style="color:purple;">&#9679;</span> {POI_TYPE} (nearby)')
    legend = ('<div style="position:fixed; top:30px; right:30px; z-index:1000; '
              'background:white; border:2px solid gray; border-radius:8px; '
              'padding:12px 16px; font-family:Arial; font-size:12px; '
              'box-shadow:2px 2px 6px rgba(0,0,0,0.3); max-width:380px;">'
              '<b>Route legend</b><br><br>' + items +
              '<br><br><span style="color:green;">&#9658;</span> Origin &nbsp; '
              '<span style="color:red;">&#9632;</span> Destination</div>')
    m.get_root().html.add_child(folium.Element(legend))

    map_path = os.path.abspath(f"{OUTPUT_DIR}/route_map.html")
    m.save(map_path)
    print(f"Map saved: {map_path}")
    if OPEN_IN_BROWSER:
        webbrowser.open("file:///" + map_path.replace("\\", "/"))
