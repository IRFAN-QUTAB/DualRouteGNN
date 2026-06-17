import argparse
import pandas as pd
from neo4j import GraphDatabase
import time
import math


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_roadosm_osmway_ids(self):
        """Get all osmway_ids from RoadOsm nodes"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:RoadOsm)
                RETURN n.osmway_id AS osmway_id
            """)
            return [record['osmway_id'] for record in result]

    def set_aadt(self, osmway_id, aadt):
        """Set AADT on a single RoadOsm node"""
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run("""
                    MATCH (n:RoadOsm {osmway_id: $osmway_id})
                    SET n.AADT = $aadt
                """, osmway_id=osmway_id, aadt=aadt).consume()
            )

    def set_aadt_batch(self, batch):
        """Set AADT on multiple RoadOsm nodes in one transaction"""
        with self.driver.session() as session:
            session.execute_write(self._set_aadt_batch, batch)

    @staticmethod
    def _set_aadt_batch(tx, batch):
        tx.run("""
            UNWIND $batch AS row
            MATCH (n:RoadOsm {osmway_id: row.osmway_id})
            SET n.AADT = row.aadt
        """, batch=batch)

    def verify(self):
        """Verify AADT integration"""
        with self.driver.session() as session:
            total = session.run(
                "MATCH (n:RoadOsm) RETURN count(n) AS cnt"
            ).single()['cnt']

            with_aadt = session.run(
                "MATCH (n:RoadOsm) WHERE n.AADT IS NOT NULL RETURN count(n) AS cnt"
            ).single()['cnt']

            without_aadt = session.run(
                "MATCH (n:RoadOsm) WHERE n.AADT IS NULL RETURN count(n) AS cnt"
            ).single()['cnt']

            stats = session.run("""
                MATCH (n:RoadOsm)
                WHERE n.AADT IS NOT NULL
                RETURN min(n.AADT) AS min_aadt,
                       max(n.AADT) AS max_aadt,
                       avg(n.AADT) AS avg_aadt
            """).single()

            sample = session.run("""
                MATCH (n:RoadOsm)
                WHERE n.AADT IS NOT NULL
                RETURN n.osmway_id AS osmway_id,
                       n.name AS name,
                       n.highway AS highway,
                       n.AADT AS aadt
                ORDER BY n.AADT DESC
                LIMIT 5
            """).values()

            highway_coverage = session.run("""
                MATCH (n:RoadOsm)
                WITH n.highway AS highway,
                     count(n) AS total,
                     sum(CASE WHEN n.AADT IS NOT NULL THEN 1 ELSE 0 END) AS with_aadt
                RETURN highway, total, with_aadt,
                       round(100.0 * with_aadt / total, 1) AS pct
                ORDER BY total DESC
            """).values()

        print(f"\n  Total RoadOsm nodes:    {total:,}")
        print(f"  With AADT:              {with_aadt:,}")
        print(f"  Without AADT:           {without_aadt:,}")
        print(f"  Coverage:               {100.0 * with_aadt / total:.2f}%")

        if stats['min_aadt'] is not None:
            print(f"\n  AADT stats:")
            print(f"    Min:  {stats['min_aadt']:.2f}")
            print(f"    Max:  {stats['max_aadt']:.2f}")
            print(f"    Mean: {stats['avg_aadt']:.2f}")

        print(f"\n  Top 5 roads by AADT:")
        for row in sample:
            print(f"    {row[1]:40s} | {row[2]:15s} | AADT={row[3]:.0f}")

        print(f"\n  Coverage by highway type:")
        print(f"    {'Highway':20s} {'Total':>8s} {'With AADT':>10s} {'%':>8s}")
        print(f"    {'-' * 20} {'-' * 8} {'-' * 10} {'-' * 8}")
        for row in highway_coverage:
            print(f"    {str(row[0]):20s} {row[1]:8,} {row[2]:10,} {row[3]:7.1f}%")


def add_options():
    parser = argparse.ArgumentParser(description='Integrate AADT traffic data into dual graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--csv', '-c', dest='csv_path', type=str, required=True,
                        help='Path to Madrid_AADT_clean.csv')
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    # ===========================================
    # STEP 1: Load CSV
    # ===========================================
    print("\n=== Step 1: Loading CSV ===")
    df = pd.read_csv(options.csv_path)
    print(f"  Total rows in CSV: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Convert osmid from float to string (to match Neo4j osmway_id)
    df['osmid_str'] = df['osmid'].apply(
        lambda x: str(int(x)) if pd.notna(x) and not math.isnan(x) else None
    )
    df = df.dropna(subset=['osmid_str'])
    print(f"  Valid osmid rows: {len(df)}")

    # ===========================================
    # STEP 2: Get RoadOsm osmway_ids from Neo4j
    # ===========================================
    print("\n=== Step 2: Getting RoadOsm osmway_ids from Neo4j ===")
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    neo4j_ids = set(greeter.get_roadosm_osmway_ids())
    print(f"  RoadOsm nodes in Neo4j: {len(neo4j_ids):,}")

    # ===========================================
    # STEP 3: Match CSV osmids with Neo4j
    # ===========================================
    print("\n=== Step 3: Matching CSV with Neo4j ===")

    # Find matches
    df['matched'] = df['osmid_str'].isin(neo4j_ids)
    matched_df = df[df['matched']].copy()
    unmatched_df = df[~df['matched']].copy()

    print(f"  CSV rows matched:     {len(matched_df):,}")
    print(f"  CSV rows not matched: {len(unmatched_df):,}")

    # Handle duplicates — same osmid multiple sensors, take average AADT
    matched_agg = matched_df.groupby('osmid_str').agg(
        AADT=('AADT', 'mean')
    ).reset_index()
    print(f"  Unique segments with AADT: {len(matched_agg):,}")

    # ===========================================
    # STEP 4: Write AADT to Neo4j in batches
    # ===========================================
    print("\n=== Step 4: Writing AADT to Neo4j ===")

    batch_size = 500
    total_written = 0
    records = matched_agg.to_dict('records')

    for i in range(0, len(records), batch_size):
        batch = []
        for row in records[i:i + batch_size]:
            batch.append({
                'osmway_id': row['osmid_str'],
                'aadt': float(row['AADT'])
            })
        greeter.set_aadt_batch(batch)
        total_written += len(batch)
        print(f"  Written: {total_written:,} / {len(records):,}")

    print(f"  Total AADT values written: {total_written:,}")

    # ===========================================
    # STEP 5: Verify
    # ===========================================
    print("\n=== Step 5: Verification ===")
    greeter.verify()

    greeter.close()
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return 0


main()
