import argparse
from neo4j import GraphDatabase
import time


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def check_graph(self):
        """Check RoadOsm nodes and CONNECTED edges exist"""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n:RoadOsm) RETURN count(n) AS cnt").single()['cnt']
            edges = session.run("MATCH ()-[r:CONNECTED]->() RETURN count(r) AS cnt").single()['cnt']
        print(f"  RoadOsm nodes: {nodes:,}")
        print(f"  CONNECTED edges: {edges:,}")
        return nodes, edges

    def drop_projection(self, name):
        """Drop graph projection if it already exists"""
        with self.driver.session() as session:
            exists = session.run(
                "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                name=name
            ).single()['exists']
            if exists:
                session.run("CALL gds.graph.drop($name)", name=name)
                print(f"  Dropped existing projection: {name}")

    def create_projection(self):
        """Create GDS graph projection for RoadOsm + CONNECTED (Neo4j 4.x syntax)"""
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.graph.create(
                    'namebased_dual',
                    'RoadOsm',
                    {
                        CONNECTED: {
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
            """).single()
        print(f"  Projection: {result['graphName']}")
        print(f"  Nodes: {result['nodeCount']:,}")
        print(f"  Relationships: {result['relationshipCount']:,}")

    def compute_pagerank(self):
        """Compute PageRank and write to nodes"""
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.pageRank.write(
                    'namebased_dual',
                    {
                        maxIterations: 20,
                        dampingFactor: 0.85,
                        writeProperty: 'pagerank'
                    }
                )
                YIELD nodePropertiesWritten, ranIterations, didConverge,
                      centralityDistribution
                RETURN nodePropertiesWritten, ranIterations, didConverge,
                       centralityDistribution
            """).single()
        print(f"  Nodes written: {result['nodePropertiesWritten']:,}")
        print(f"  Iterations: {result['ranIterations']}")
        print(f"  Converged: {result['didConverge']}")
        dist = result['centralityDistribution']
        print(f"  Min: {dist['min']:.6f}, Max: {dist['max']:.6f}, Mean: {dist['mean']:.6f}")

    def compute_betweenness(self):
        """Compute exact Betweenness Centrality and write to nodes"""
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.betweenness.write(
                    'namebased_dual',
                    {
                        writeProperty: 'betweenness'
                    }
                )
                YIELD nodePropertiesWritten
                RETURN nodePropertiesWritten
            """).single()
            print(f"  Nodes written: {result['nodePropertiesWritten']:,}")

        with self.driver.session() as session:
            stats = session.run("""
                MATCH (n:RoadOsm)
                WHERE n.betweenness IS NOT NULL
                RETURN min(n.betweenness) AS min_bc,
                       max(n.betweenness) AS max_bc,
                       avg(n.betweenness) AS avg_bc,
                       count(n) AS total
            """).single()
            print(f"  Total: {stats['total']:,}")
            print(f"  Min: {stats['min_bc']:.2f}, Max: {stats['max_bc']:.2f}, Mean: {stats['avg_bc']:.2f}")

    def compute_degree(self):
        """Compute Degree Centrality and write to nodes"""
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.degree.write(
                    'namebased_dual',
                    {
                        writeProperty: 'degree'
                    }
                )
                YIELD nodePropertiesWritten, centralityDistribution
                RETURN nodePropertiesWritten, centralityDistribution
            """).single()
        print(f"  Nodes written: {result['nodePropertiesWritten']:,}")
        dist = result['centralityDistribution']
        print(f"  Min: {dist['min']:.6f}, Max: {dist['max']:.6f}, Mean: {dist['mean']:.6f}")

    def verify(self):
        """Verify all centrality properties are written"""
        with self.driver.session() as session:
            sample = session.run("""
                MATCH (n:RoadOsm)
                WHERE n.name IS NOT NULL
                RETURN n.name AS name,
                       n.highway AS highway,
                       n.pagerank AS pagerank,
                       n.betweenness AS betweenness,
                       n.degree AS degree
                ORDER BY n.pagerank DESC
                LIMIT 5
            """).values()

            nulls = session.run("""
                MATCH (n:RoadOsm)
                RETURN 
                    sum(CASE WHEN n.pagerank IS NULL THEN 1 ELSE 0 END) AS pr_null,
                    sum(CASE WHEN n.betweenness IS NULL THEN 1 ELSE 0 END) AS bc_null,
                    sum(CASE WHEN n.degree IS NULL THEN 1 ELSE 0 END) AS deg_null
            """).single()

        print(f"\n  Null check:")
        print(f"    PageRank nulls: {nulls['pr_null']}")
        print(f"    Betweenness nulls: {nulls['bc_null']}")
        print(f"    Degree nulls: {nulls['deg_null']}")

        print(f"\n  Top 5 roads by PageRank:")
        for row in sample:
            print(f"    {row[0]:30s} | {row[1]:15s} | PR={row[2]:.6f} | BC={row[3]:.2f} | Deg={row[4]:.0f}")


def add_options():
    parser = argparse.ArgumentParser(description='Compute centrality on segment-level dual graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)

    # Step 1: Check graph
    print("\n=== Step 1: Checking dual graph ===")
    nodes, edges = greeter.check_graph()
    if nodes == 0:
        print("  ERROR: No RoadOsm nodes found! Run Step 1 first.")
        greeter.close()
        return 1

    # Step 2: Create GDS projection
    print("\n=== Step 2: Creating GDS projection ===")
    greeter.drop_projection('namebased_dual')
    greeter.create_projection()

    # Step 3: PageRank
    print("\n=== Step 3: Computing PageRank (damping=0.85, maxIter=20) ===")
    t1 = time.time()
    greeter.compute_pagerank()
    print(f"  Time: {time.time() - t1:.1f}s")

    # Step 4: Betweenness Centrality
    print("\n=== Step 4: Computing Betweenness Centrality ===")
    print("  (This may take several minutes on large graphs...)")
    t2 = time.time()
    greeter.compute_betweenness()
    print(f"  Time: {time.time() - t2:.1f}s")

    # Step 5: Degree Centrality
    print("\n=== Step 5: Computing Degree Centrality ===")
    t3 = time.time()
    greeter.compute_degree()
    print(f"  Time: {time.time() - t3:.1f}s")

    # Step 6: Clean up projection
    print("\n=== Step 6: Cleaning up projection ===")
    greeter.drop_projection('namebased_dual')
    print("  Projection dropped.")

    # Step 7: Verify
    print("\n=== Step 7: Verification ===")
    greeter.verify()

    greeter.close()

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    return 0


main()
