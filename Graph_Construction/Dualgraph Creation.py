import argparse
from neo4j import GraphDatabase
import time


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clean_previous(self):
        """Remove previous RoadOsm nodes if any"""
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(
                "MATCH (n:RoadOsm) DETACH DELETE n"
            ).consume())
        print("Previous RoadOsm nodes cleaned.")

    def create_index(self):
        """Create index on RoadOsm for faster lookups"""
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(
                "CREATE INDEX IF NOT EXISTS FOR (r:RoadOsm) ON (r.osmway_id)"
            ).consume())
        print("Index on RoadOsm.osmway_id created.")

    def create_nodes(self):
        """Create dual graph nodes — one per osmway_id with all properties"""
        with self.driver.session() as session:
            result = session.execute_write(self._create_nodes)
            return result

    @staticmethod
    def _create_nodes(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                "MATCH (m:RoadJunction)-[r:ROUTE {status: 'active'}]->(n:RoadJunction)
                 WITH DISTINCT r.osmway_id AS street_name
                 RETURN street_name",

                "WITH street_name
                 CREATE (road:RoadOsm {osmway_id: street_name})
                 WITH street_name
                 MATCH (m:RoadJunction)-[r1:ROUTE {osmway_id: street_name, status: 'active'}]->(n:RoadJunction)
                 WITH sum(r1.distance) AS dist, street_name, r1.name AS road_name,
                      r1.highway AS highway, r1.oneway AS oneway, r1.lanes AS lanes,
                      r1.maxspeed AS maxspeed, r1.bridge AS bridge, r1.tunnel AS tunnel,
                      r1.width AS width, r1.service AS service, r1.access AS access,
                      r1.junction AS junction_type, r1.ref AS ref
                 MATCH (d:RoadOsm {osmway_id: street_name})
                 SET d.status = 'active',
                     d.distance = dist,
                     d.name = road_name,
                     d.highway = highway,
                     d.oneway = oneway,
                     d.lanes = lanes,
                     d.maxspeed = maxspeed,
                     d.bridge = bridge,
                     d.tunnel = tunnel,
                     d.width = width,
                     d.service = service,
                     d.access = access,
                     d.junction_type = junction_type,
                     d.ref = ref",

                {batchSize: 1000, iterateList: true, parallel: false}
            )
            YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def create_edges(self):
        """Create dual graph edges — connection through shared junctions"""
        with self.driver.session() as session:
            result = session.execute_write(self._create_edges)
            return result

    @staticmethod
    def _create_edges(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
                "MATCH (m:RoadJunction)-[r:ROUTE]->(n:RoadJunction)
                 WITH DISTINCT r.osmway_id AS street_name
                 RETURN street_name",

                "WITH street_name
                 MATCH (m:RoadJunction)-[r1:ROUTE {osmway_id: street_name}]->(n:RoadJunction)
                 WITH m, street_name
                 MATCH (x:RoadJunction)-[r2:ROUTE]->(m:RoadJunction)
                 WHERE r2.osmway_id <> street_name
                 WITH r2.osmway_id AS source, street_name, m
                 MATCH (r1:RoadOsm {osmway_id: source}), (r2:RoadOsm {osmway_id: street_name})
                 CREATE (r1)-[r:CONNECTED {junction: m.osmnode_id, location: m.location}]->(r2)",

                {batchSize: 500, iterateList: true, parallel: false}
            )
            YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """)
        return result.values()

    def verify(self):
        """Print stats about the created dual graph"""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n:RoadOsm) RETURN count(n) AS cnt").single()['cnt']
            edges = session.run("MATCH ()-[r:CONNECTED]->() RETURN count(r) AS cnt").single()['cnt']

            sample = session.run("""
                MATCH (n:RoadOsm) 
                WHERE n.name IS NOT NULL
                RETURN n.osmway_id AS osmway_id, n.name AS name, 
                       n.highway AS highway, n.distance AS distance
                LIMIT 5
            """).values()

            props = session.run("""
                MATCH (n:RoadOsm)
                RETURN keys(n) AS props
                LIMIT 1
            """).single()['props']

        print(f"\n  RoadOsm nodes: {nodes:,}")
        print(f"  CONNECTED edges: {edges:,}")
        print(f"  Node properties: {props}")
        print(f"\n  Sample nodes:")
        for row in sample:
            print(f"    {row}")


def add_options():
    parser = argparse.ArgumentParser(description='Creation of segment-level dual graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str,
                        help="""Insert the address of the local neo4j instance. For example: neo4j://localhost:7687""",
                        required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str,
                        help="""Insert the name of the user of the local neo4j instance.""",
                        required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str,
                        help="""Insert the password of the local neo4j instance.""",
                        required=True)
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)

    # Step 0: Clean previous
    print("\n=== Step 0: Cleaning previous RoadOsm nodes ===")
    greeter.clean_previous()

    # Step 1: Create index
    print("\n=== Step 1: Creating index ===")
    greeter.create_index()

    # Step 2: Create nodes
    print("\n=== Step 2: Creating dual graph nodes (batch size: 1000) ===")
    t1 = time.time()
    node_result = greeter.create_nodes()
    print(f"  Result: {node_result}")
    print(f"  Time: {time.time() - t1:.1f}s")

    # Step 3: Create edges
    print("\n=== Step 3: Creating dual graph edges (batch size: 500) ===")
    t2 = time.time()
    edge_result = greeter.create_edges()
    print(f"  Result: {edge_result}")
    print(f"  Time: {time.time() - t2:.1f}s")

    # Step 4: Verify
    print("\n=== Step 4: Verification ===")
    greeter.verify()

    greeter.close()

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return 0


main()
