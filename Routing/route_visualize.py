# =============================================================================
# CELL 7: MAP VISUALIZATION 
# =============================================================================
import math
import folium, webbrowser, os

src_pos = node_positions[SOURCE_NODE]
tgt_pos = node_positions[TARGET_NODE]
clat = (src_pos[0] + tgt_pos[0]) / 2
clon = (src_pos[1] + tgt_pos[1]) / 2

m = folium.Map(location=[clat, clon], zoom_start=14)

# Draw routes
for route in routes:
    coords = [node_positions[n] for n in route['path'] if n in node_positions]
    if not coords: continue
    poi_info = ", ".join(f"{k}={v}" for k, v in route['poi_individual'].items() if v > 0)
    if not poi_info: poi_info = "none"
    popup_html = "<b>" + route['label'] + "</b><br><i>" + route['algo'] + "</i><hr>"
    popup_html += f"Distance: {route['distance_m']:.0f}m<br>"
    avg_traffic = route['traffic_exposure'] / max(route['distance_m'], 1)
    popup_html += f"Avg Traffic: {avg_traffic:.0f} vehicles/day<br>"
    popup_html += f"Roads: {route['roads_visited']}<br>"
    popup_html += "POIs: " + str(route['poi_total']) + " (" + poi_info + ")"
    line = folium.PolyLine(coords, color=route['color'], weight=route['weight'], opacity=0.8,
        popup=folium.Popup(popup_html, max_width=350),
        tooltip=route['label'] + ": " + f"{route['distance_m']:.0f}m")
    if route['dash']: line.options['dashArray'] = route['dash']
    line.add_to(m)

# PROFESSOR POINT 3: Mark actual POI locations ON the routes (purple)
if POI_TYPE:
    pois_of_type = df_pois[df_pois['type'] == POI_TYPE]
    on_route_marked = 0
    on_route_poi_coords = []  # remember to skip in nearby pass

    for _, poi in pois_of_type.iterrows():
        if pd.notna(poi['lat']) and pd.notna(poi['lon']):
            poi_lat, poi_lon = float(poi['lat']), float(poi['lon'])
            near_route = False
            for route in routes:
                for n in route['path']:
                    if n in node_positions:
                        n_lat, n_lon = node_positions[n]
                        dist = math.sqrt((poi_lat - n_lat)**2 + (poi_lon - n_lon)**2) * 111000
                        if dist < 100:
                            near_route = True
                            break
                if near_route: break
            if near_route:
                poi_name = poi['name'] if pd.notna(poi['name']) else POI_TYPE
                folium.Marker(
                    [poi_lat, poi_lon],
                    popup=f"<b>{POI_TYPE}</b><br>{poi_name}",
                    icon=folium.Icon(color='purple', icon='plus-sign', prefix='glyphicon')
                ).add_to(m)
                on_route_poi_coords.append((poi_lat, poi_lon))
                on_route_marked += 1
    print(f"POIs on route ({POI_TYPE}): {on_route_marked}")

    # PROFESSOR POINT 4: Mark ALL nearby POIs in a wider area (gray)
    all_lats, all_lons = [], []
    for route in routes:
        for n in route['path']:
            if n in node_positions:
                lat, lon = node_positions[n]
                all_lats.append(lat); all_lons.append(lon)

    if all_lats:
        pad = 0.001  # ~3 km padding (was 0.005 = 500m)
        min_lat, max_lat = min(all_lats) - pad, max(all_lats) + pad
        min_lon, max_lon = min(all_lons) - pad, max(all_lons) + pad

        nearby_pois = df_pois[
            (df_pois['type'] == POI_TYPE) &
            (df_pois['lat'] >= min_lat) & (df_pois['lat'] <= max_lat) &
            (df_pois['lon'] >= min_lon) & (df_pois['lon'] <= max_lon)
        ]

        nearby_count = 0
        for _, poi in nearby_pois.iterrows():
            if pd.notna(poi['lat']) and pd.notna(poi['lon']):
                poi_lat, poi_lon = float(poi['lat']), float(poi['lon'])

                # Skip if already marked as on-route (purple)
                already_marked = False
                for (rlat, rlon) in on_route_poi_coords:
                    if abs(rlat - poi_lat) < 1e-6 and abs(rlon - poi_lon) < 1e-6:
                        already_marked = True
                        break

                if not already_marked:
                        poi_name = poi['name'] if pd.notna(poi['name']) else POI_TYPE

                        folium.CircleMarker(
                            [poi_lat, poi_lon],
                            radius=3,
                            color='purple',
                            fill=True,
                            fill_color='purple',
                            fill_opacity=0.7,
                            popup=f"<b>{POI_TYPE} (nearby)</b><br>{poi_name}"
                        ).add_to(m)

                        nearby_count += 1
        print(f"Nearby {POI_TYPE} (not on route): {nearby_count}")

# Origin & Destination markers
folium.Marker(src_pos, popup="<b>ORIGIN</b>",
              icon=folium.Icon(color='green', icon='play', prefix='fa')).add_to(m)
folium.Marker(tgt_pos, popup="<b>DESTINATION</b>",
              icon=folium.Icon(color='red', icon='stop', prefix='fa')).add_to(m)

# Highlight waypoint POI edge (if exists)
if POI_TYPE and 'WAYPOINT' in dir() and WAYPOINT:
    wp_u, wp_v = WAYPOINT
    if wp_u in node_positions and wp_v in node_positions:
        folium.PolyLine(
            [node_positions[wp_u], node_positions[wp_v]],
            color='#FF00FF', weight=10, opacity=0.6,
            tooltip=f"Waypoint POI edge ({POI_TYPE})"
        ).add_to(m)

# Legend
legend_items = ""
for r in routes:
    algo_name = r['algo']
    legend_items += '<span style="color:' + r["color"] + '; font-size:16px;">&#9644;&#9644;</span>'
    legend_items += '&nbsp; <b>' + r["label"] + '</b> — ' + f"{r['distance_m']:.0f}m ({algo_name})<br>"
if POI_TYPE:
    legend_items += '<br><span style="color:purple;">&#10010;</span> ' + POI_TYPE + ' (on route)'
    legend_items += '<br><span style="color:purple;">●</span> ' + POI_TYPE + ' (nearby, not on route)'
legend_div = '<div style="position:fixed; top:30px; right:30px; z-index:1000; '
legend_div += 'background:white; border:2px solid gray; border-radius:8px; '
legend_div += 'padding:12px 16px; font-family:Arial; font-size:12px; '
legend_div += 'box-shadow:2px 2px 6px rgba(0,0,0,0.3); max-width:350px;">'
legend_div += '<b>Route Legend</b><br><br>' + legend_items
legend_div += '<br><br><span style="color:green;">&#9658;</span> Origin &nbsp; '
legend_div += '<span style="color:red;">&#9632;</span> Destination</div>'
m.get_root().html.add_child(folium.Element(legend_div))

map_path = os.path.abspath(f"{OUTPUT_DIR}/route_map_interactive.html")
m.save(map_path)
webbrowser.open("file:///" + map_path)
print("Map opened: " + map_path)
