import os
import sys
import warnings
import pandas as pd
from neo4j import GraphDatabase

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["NEO4J_NOTIFICATIONS_ENABLED"] = "false"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "irfan1234"

OUTPUT_DIR = r"D:\Manageable\MY Data\roadRouting New\Madrid\step1\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------
# POI CATEGORY MAPPING
# -----------------------------------------------------------------
CATEGORY_MAP = {
    'hospital': 'health', 'doctors': 'health', 'pharmacy': 'health',
    'clinic': 'health', 'dentist': 'health', 'veterinary': 'health',
    'nursing_home': 'health',
    'school': 'education', 'university': 'education', 'college': 'education',
    'kindergarten': 'education', 'library': 'education',
    'driving_school': 'education', 'language_school': 'education',
    'music_school': 'education', 'library_dropoff': 'education',
    'restaurant': 'commercial', 'cafe': 'commercial', 'fast_food': 'commercial',
    'bar': 'commercial', 'pub': 'commercial', 'bank': 'commercial',
    'supermarket': 'commercial', 'convenience': 'commercial',
    'clothes': 'commercial', 'shop': 'commercial', 'bakery': 'commercial',
    'butcher': 'commercial', 'greengrocer': 'commercial',
    'hairdresser': 'commercial', 'beauty_shop': 'commercial',
    'optician': 'commercial', 'shoes': 'commercial',
    'department_store': 'commercial', 'mall': 'commercial',
    'stationery': 'commercial', 'bookshop': 'commercial',
    'newsagent': 'commercial', 'kiosk': 'commercial',
    'tobacco': 'commercial', 'florist': 'commercial',
    'ice_cream': 'commercial', 'confectionery': 'commercial',
    'marketplace': 'commercial', 'travel_agent': 'commercial',
    'laundry': 'commercial', 'dry_cleaning': 'commercial',
    'furniture_shop': 'commercial', 'hardware': 'commercial',
    'electronics': 'commercial', 'mobile_phone_shop': 'commercial',
    'gift_shop': 'commercial', 'jeweller': 'commercial',
    'hotel': 'commercial', 'guest_house': 'commercial',
    'hostel': 'commercial', 'nightclub': 'commercial',
    'biergarten': 'commercial',
    'atm': 'commercial', 'vending_machine': 'commercial',
    'money_transfer': 'commercial', 'animal_training': 'commercial',
    'food_court': 'commercial', 'gambling': 'commercial',
    'conference_centre': 'commercial', 'casino': 'commercial',
    'events_venue': 'commercial',
    'place_of_worship': 'public_services', 'church': 'public_services',
    'fire_station': 'public_services', 'police': 'public_services',
    'post_office': 'public_services', 'townhall': 'public_services',
    'courthouse': 'public_services', 'community_centre': 'public_services',
    'theatre': 'public_services', 'cinema': 'public_services',
    'museum': 'public_services', 'arts_centre': 'public_services',
    'public_building': 'public_services', 'social_facility': 'public_services',
    'recycling': 'public_services', 'cemetery': 'public_services',
    'bench': 'public_services', 'waste_disposal': 'public_services',
    'drinking_water': 'public_services', 'waste_basket': 'public_services',
    'fountain': 'public_services', 'toilets': 'public_services',
    'telephone': 'public_services', 'post_box': 'public_services',
    'shelter': 'public_services', 'monastery': 'public_services',
    'grave_yard': 'public_services', 'archive': 'public_services',
    'animal_shelter': 'public_services',
    'parking': 'transport', 'fuel': 'transport',
    'bus_station': 'transport', 'bus_stop': 'transport',
    'taxi': 'transport', 'car_wash': 'transport',
    'bicycle_rental': 'transport', 'car_rental': 'transport',
    'car_sharing': 'transport', 'charging_station': 'transport',
    'bicycle_parking': 'transport',
    'parking_entrance': 'transport', 'parking_space': 'transport',
    'bicycle_repair_station': 'transport', 'motorcycle_parking': 'transport',
    'sanitary_dump_station': 'transport',
}

CATEGORIES = ['health', 'education', 'commercial', 'public_services', 'transport']


# -----------------------------------------------------------------
# CONNECT
# -----------------------------------------------------------------
print("=" * 60)
print("DualRouteGNN - Step 4: Neo4j to CSV Export (Segment-Level)")
print("=" * 60)

print("\nConnecting to Neo4j...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")
    print("  Connected!")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)


def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]


# =============================================================
# STEP 0: VERIFY SCHEMA
# =============================================================
print("\n[0] Verifying schema...")

roadosm_count = run_query("MATCH (n:RoadOsm) RETURN count(n) AS cnt")
print(f"  RoadOsm nodes: {roadosm_count[0]['cnt']:,}")

connected_count = run_query("MATCH ()-[r:CONNECTED]->() RETURN count(r) AS cnt")
print(f"  CONNECTED edges: {connected_count[0]['cnt']:,}")

near_fwd = run_query("MATCH (p:POI)-[:NEAR]->(rj:RoadJunction) RETURN count(*) as cnt")
print(f"  (POI)-[:NEAR]->(RoadJunction): {near_fwd[0]['cnt']:,}")

tag_count = run_query("MATCH (p:POI)-[:TAGS]->(t:Tag) WHERE t.amenity IS NOT NULL RETURN count(*) as cnt")
print(f"  POIs with Tag.amenity: {tag_count[0]['cnt']:,}")


# =============================================================
# EXPORT 1: DUAL GRAPH NODES (RoadOsm) - Segment Level
# =============================================================
print("\n" + "=" * 60)
print("[1/7] Exporting DUAL GRAPH nodes (RoadOsm - segment level)...")
print("=" * 60)

dual_nodes = run_query("""
    MATCH (road:RoadOsm)
    WHERE road.status = 'active'
    RETURN road.osmway_id AS osmway_id,
           road.name AS name,
           COALESCE(road.AADT, 0.0) AS AADT,
           COALESCE(toFloat(road.distance), 0.0) AS distance,
           COALESCE(road.pagerank, 0.0) AS pagerank,
           COALESCE(road.betweenness, 0.0) AS betweenness,
           COALESCE(road.degree, 0.0) AS degree,
           COALESCE(road.highway, 'unknown') AS highway,
           COALESCE(road.oneway, 'false') AS oneway,
           road.lanes AS lanes,
           road.maxspeed AS maxspeed
    ORDER BY road.osmway_id
""")

df_dual_nodes = pd.DataFrame(dual_nodes)
df_dual_nodes['node_id'] = range(len(df_dual_nodes))

print(f"  Found {len(df_dual_nodes):,} RoadOsm nodes")
print(f"  With AADT > 0: {(df_dual_nodes['AADT'] > 0).sum():,}")
print(f"  AADT range: {df_dual_nodes['AADT'].min():.1f} - {df_dual_nodes['AADT'].max():.1f}")

df_dual_nodes.to_csv(f"{OUTPUT_DIR}/dual_nodes.csv", index=False)
print(f"  Saved -> dual_nodes.csv")


# =============================================================
# EXPORT 2: DUAL GRAPH EDGES (CONNECTED)
# =============================================================
print("\n" + "=" * 60)
print("[2/7] Exporting DUAL GRAPH edges (CONNECTED)...")
print("=" * 60)

dual_edges = run_query("""
    MATCH (r1:RoadOsm)-[c:CONNECTED]->(r2:RoadOsm)
    WHERE r1.status = 'active' AND r2.status = 'active'
    RETURN r1.osmway_id AS source_osmway_id,
           r2.osmway_id AS target_osmway_id
""")

df_dual_edges = pd.DataFrame(dual_edges)

osmway_to_id = dict(zip(df_dual_nodes['osmway_id'], df_dual_nodes['node_id']))
df_dual_edges['source_id'] = df_dual_edges['source_osmway_id'].map(osmway_to_id)
df_dual_edges['target_id'] = df_dual_edges['target_osmway_id'].map(osmway_to_id)

before = len(df_dual_edges)
df_dual_edges = df_dual_edges.dropna(subset=['source_id', 'target_id'])
df_dual_edges['source_id'] = df_dual_edges['source_id'].astype(int)
df_dual_edges['target_id'] = df_dual_edges['target_id'].astype(int)

print(f"  Found {len(df_dual_edges):,} CONNECTED edges")
if before != len(df_dual_edges):
    print(f"  Dropped {before - len(df_dual_edges)} unmapped edges")

df_dual_edges.to_csv(f"{OUTPUT_DIR}/dual_edges.csv", index=False)
print(f"  Saved -> dual_edges.csv")


# =============================================================
# EXPORT 3: POI FEATURES - GROUPED (for GNN)
# =============================================================
print("\n" + "=" * 60)
print("[3/7] Exporting GROUPED POI features (for GNN)...")
print("=" * 60)

poi_query = """
    MATCH (junction:RoadJunction)-[route:ROUTE]-(other:RoadJunction)
    WHERE route.osmway_id IS NOT NULL AND route.status = 'active'
    WITH route.osmway_id AS segment_id, junction
    WITH DISTINCT segment_id, junction
    MATCH (poi:POI)-[:NEAR]->(junction)
    MATCH (poi)-[:TAGS]->(t:Tag)
    WHERE t.amenity IS NOT NULL
    RETURN segment_id, t.amenity AS poi_type, count(DISTINCT poi) AS cnt
    ORDER BY segment_id, cnt DESC
"""

print("  Running POI query...")
poi_data = run_query(poi_query)
df_poi_raw = pd.DataFrame(poi_data)

if len(df_poi_raw) > 0:
    print(f"  Found {len(df_poi_raw):,} segment-POI combinations")
    print(f"  Segments with POIs: {df_poi_raw['segment_id'].nunique():,}")

    df_poi_pivot = df_poi_raw.pivot_table(
        index='segment_id', columns='poi_type', values='cnt', fill_value=0
    ).reset_index()

    poi_types_found = [c for c in df_poi_pivot.columns if c != 'segment_id']

    for cat in CATEGORIES:
        df_poi_pivot[cat] = 0
    for pt in poi_types_found:
        cat = CATEGORY_MAP.get(pt, None)
        if cat and cat in CATEGORIES:
            df_poi_pivot[cat] += df_poi_pivot[pt]

    print(f"\n  {'Category':20s} {'Total POIs':>10s} {'Segments':>10s}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10}")
    for cat in CATEGORIES:
        total = int(df_poi_pivot[cat].sum())
        segs = int((df_poi_pivot[cat] > 0).sum())
        print(f"  {cat:20s} {total:10d} {segs:10d}")

    poi_grouped = df_poi_pivot[['segment_id'] + CATEGORIES]
else:
    print("  WARNING: No POI-segment connections found!")
    poi_grouped = pd.DataFrame(columns=['segment_id'] + CATEGORIES)

poi_grouped.to_csv(f"{OUTPUT_DIR}/poi_features_grouped.csv", index=False)
print(f"  Saved -> poi_features_grouped.csv")


# =============================================================
# EXPORT 4: POI FEATURES - INDIVIDUAL (for Routing)
# =============================================================
print("\n" + "=" * 60)
print("[4/7] Exporting INDIVIDUAL POI features (for Routing)...")
print("=" * 60)

if len(df_poi_raw) > 0:
    df_poi_individual = df_poi_raw.pivot_table(
        index='segment_id', columns='poi_type', values='cnt', fill_value=0
    ).reset_index()

    individual_types = [c for c in df_poi_individual.columns if c != 'segment_id']
    print(f"  Individual POI types found: {len(individual_types)}")
    print(f"  Segments with individual POI data: {len(df_poi_individual):,}")

    print(f"\n  {'POI Type':<30s} {'Total':>6s} {'Segs':>6s} {'Group':>15s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*15}")
    for pt in sorted(individual_types, key=lambda x: -df_poi_individual[x].sum()):
        total = int(df_poi_individual[pt].sum())
        segs = int((df_poi_individual[pt] > 0).sum())
        group = CATEGORY_MAP.get(pt, 'other')
        print(f"  {pt:<30s} {total:>6d} {segs:>6d} {group:>15s}")
else:
    df_poi_individual = pd.DataFrame(columns=['segment_id'])
    individual_types = []

df_poi_individual.to_csv(f"{OUTPUT_DIR}/poi_features_individual.csv", index=False)
print(f"\n  Saved -> poi_features_individual.csv")


# =============================================================
# EXPORT 5: ENRICHED DUAL NODES (nodes + grouped POI merged)
# =============================================================
print("\n" + "=" * 60)
print("[5/7] Creating enriched dual nodes...")
print("=" * 60)

df_dual_enriched = df_dual_nodes.merge(
    poi_grouped, left_on='osmway_id', right_on='segment_id', how='left'
).fillna(0)
if 'segment_id' in df_dual_enriched.columns:
    df_dual_enriched = df_dual_enriched.drop(columns=['segment_id'])
for cat in CATEGORIES:
    if cat in df_dual_enriched.columns:
        df_dual_enriched[cat] = df_dual_enriched[cat].astype(int)

print(f"  Enriched nodes: {len(df_dual_enriched):,}")
print(f"  Columns: {list(df_dual_enriched.columns)}")

# Show how many segments have POIs
has_poi = (df_dual_enriched[CATEGORIES].sum(axis=1) > 0).sum()
print(f"  Segments with at least 1 POI: {has_poi:,} / {len(df_dual_enriched):,} ({100*has_poi/len(df_dual_enriched):.1f}%)")

df_dual_enriched.to_csv(f"{OUTPUT_DIR}/dual_nodes_enriched.csv", index=False)
print(f"  Saved -> dual_nodes_enriched.csv")


# =============================================================
# EXPORT 6: PRIMAL GRAPH NODES (RoadJunction)
# =============================================================
print("\n" + "=" * 60)
print("[6/7] Exporting PRIMAL GRAPH nodes (RoadJunction)...")
print("=" * 60)

primal_nodes = run_query("""
    MATCH (j:RoadJunction)
    OPTIONAL MATCH (poi:POI)-[:NEAR]->(j)
    WITH j, count(DISTINCT poi) AS poi_count
    RETURN j.osmnode_id AS osmnode_id,
           j.lat AS lat,
           j.lon AS lon,
           COALESCE(j.driveable, false) AS driveable,
           poi_count
    ORDER BY j.osmnode_id
""")

df_primal_nodes = pd.DataFrame(primal_nodes)
df_primal_nodes['node_id'] = range(len(df_primal_nodes))

print(f"  Found {len(df_primal_nodes):,} RoadJunction nodes")
print(f"  Driveable: {df_primal_nodes['driveable'].sum():,}/{len(df_primal_nodes):,}")

df_primal_nodes.to_csv(f"{OUTPUT_DIR}/primal_nodes.csv", index=False)
print(f"  Saved -> primal_nodes.csv")


# =============================================================
# EXPORT 7: PRIMAL GRAPH EDGES (ROUTE)
# =============================================================
print("\n" + "=" * 60)
print("[7/7] Exporting PRIMAL GRAPH edges (ROUTE)...")
print("=" * 60)

primal_edges = run_query("""
    MATCH (j1:RoadJunction)-[r:ROUTE]->(j2:RoadJunction)
    WHERE r.status = 'active'
    RETURN j1.osmnode_id AS source_osmnode_id,
           j2.osmnode_id AS target_osmnode_id,
           COALESCE(toFloat(r.distance), 0.0) AS distance,
           COALESCE(r.highway, 'unknown') AS highway,
           COALESCE(r.name, '') AS road_name,
           r.osmway_id AS osmway_id,
           COALESCE(r.driveable, false) AS driveable,
           COALESCE(r.oneway, 'false') AS oneway
    ORDER BY j1.osmnode_id
""")

df_primal_edges = pd.DataFrame(primal_edges)

osmnode_to_id = dict(zip(df_primal_nodes['osmnode_id'], df_primal_nodes['node_id']))
df_primal_edges['source_id'] = df_primal_edges['source_osmnode_id'].map(osmnode_to_id)
df_primal_edges['target_id'] = df_primal_edges['target_osmnode_id'].map(osmnode_to_id)

before = len(df_primal_edges)
df_primal_edges = df_primal_edges.dropna(subset=['source_id', 'target_id'])
df_primal_edges['source_id'] = df_primal_edges['source_id'].astype(int)
df_primal_edges['target_id'] = df_primal_edges['target_id'].astype(int)

print(f"  Found {len(df_primal_edges):,} active ROUTE edges")
print(f"  With road_name: {(df_primal_edges['road_name'] != '').sum():,}")
print(f"  Highway types: {df_primal_edges['highway'].value_counts().head(5).to_dict()}")

df_primal_edges.to_csv(f"{OUTPUT_DIR}/primal_edges.csv", index=False)
print(f"  Saved -> primal_edges.csv")


# =============================================================
# SUMMARY
# =============================================================
driver.close()

print(f"\n{'='*60}")
print("STEP 4 COMPLETE!")
print(f"{'='*60}")
print(f"\n  Files in {OUTPUT_DIR}/:")
print(f"    dual_nodes.csv              = {len(df_dual_nodes):,} segments")
print(f"    dual_nodes_enriched.csv     = {len(df_dual_enriched):,} segments + grouped POI")
print(f"    dual_edges.csv              = {len(df_dual_edges):,} connections")
print(f"    poi_features_grouped.csv    = grouped POI (5 categories)")
print(f"    poi_features_individual.csv = individual POI ({len(individual_types)} types)")
print(f"    primal_nodes.csv            = {len(df_primal_nodes):,} junctions")
print(f"    primal_edges.csv            = {len(df_primal_edges):,} edges")
print(f"\n  Segments with real AADT: {(df_dual_nodes['AADT'] > 0).sum():,} / {len(df_dual_nodes):,}")
print(f"  Individual POI types for routing: {len(individual_types)}")
print(f"\n  Next -> Step 5 (PyTorch Geometric conversion)")
print("=" * 60)
