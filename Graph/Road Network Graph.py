import osmnx as ox
import argparse
from neo4j import GraphDatabase
import os
import time

class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def creation_graph(self, file):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_graph, file)
            return result

    @staticmethod
    def _creation_graph(tx, file):
        result = tx.run("""
                        CALL apoc.import.graphml($file, {storeNodeIds: true, defaultRelationshipType: 'ROUTE'});
                    """, file=file)
        return result.values()

    def get_path(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_path)
            return result

    @staticmethod
    def _get_path(tx):
        result = tx.run("""
                        Call dbms.listConfig() yield name,value where name = 'dbms.directories.neo4j_home' return value;
                    """)
        return result.values()

    def get_import_folder_name(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_import_folder_name)
            return result

    @staticmethod
    def _get_import_folder_name(tx):
        result = tx.run("""
                        Call dbms.listConfig() yield name,value where name = 'dbms.directories.import' return value;
                    """)
        return result.values()

    def set_label(self):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_label)
            return result

    @staticmethod
    def _creation_label(tx):
        result = tx.run("""
                        MATCH (n) SET n:RoadJunction;
                    """)
        return result.values()

    def set_location(self):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_location)
            return result

    @staticmethod
    def _creation_location(tx):
        result = tx.run("""
                           MATCH (n:RoadJunction) SET n.location = point({latitude: tofloat(n.y), longitude: tofloat(n.x)}),
                                            n.lat = tofloat(n.y), 
                                            n.lon = tofloat(n.x),
                                            n.geometry='POINT(' + n.y + ' ' + n.x +')';
                       """)
        return result.values()

    def set_distance(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_distance)
            return result

    @staticmethod
    def _set_distance(tx):
        result = tx.run("""
                           MATCH (n:RoadJunction)-[r:ROUTE]-() SET r.distance=tofloat(r.length), r.status='active'
                       """)
        return result.values()

    def rename_junction_id(self):
        with self.driver.session() as session:
            return session.execute_write(self._rename_junction_id)

    @staticmethod
    def _rename_junction_id(tx):
        return tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.id IS NOT NULL
            SET n.osmnode_id = n.id
            REMOVE n.id
        """).values()

    def rename_route_osmid(self):
        with self.driver.session() as session:
            return session.execute_write(self._rename_route_osmid)

    @staticmethod
    def _rename_route_osmid(tx):
        return tx.run("""
            MATCH ()-[r:ROUTE]->()
            WHERE r.osmid IS NOT NULL
            SET r.osmway_id = r.osmid
            REMOVE r.osmid
        """).values()

    def set_index(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_index)
            return result

    @staticmethod
    def _set_index(tx):
        return tx.run("""
            CREATE INDEX IF NOT EXISTS
            FOR (n:RoadJunction)
            ON (n.osmnode_id)
        """).values()

    def generate_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._generate_spatial_layer)
            return result

    @staticmethod
    def _generate_spatial_layer(tx):
        result = tx.run("""
                call spatial.layers()
                """)
        if len(result.values()) == 0:
            result = tx.run("""
                call spatial.addWKTLayer('spatial', 'geometry')
                """)
        return result.values()

    def import_nodes_in_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_nodes_in_spatial_layer)
            return result

    @staticmethod
    def _import_nodes_in_spatial_layer(tx):
        result = tx.run("""
        match (n:RoadJunction)
        CALL spatial.addNode('spatial', n) yield node return node;
        """)
        return result.values()

def add_options():
    parser = argparse.ArgumentParser(description='Creation of routing graph.')
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--nameFile', '-f', dest='file_name', type=str, required=True)
    return parser

def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    path = greeter.get_path()[0][0] + '\\' + greeter.get_import_folder_name()[0][0] + '\\' + options.file_name

    G = ox.graph_from_point(
        (options.lat, options.lon),
        dist=int(options.dist),
        dist_type='bbox',
        simplify=False,
        custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service"]'
    )

    # Apply custom driveability rules
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get('highway')
        access = data.get('access')

        # Default to driveable
        data['driveable'] = True

        # Apply rules
        if access in {"no", "private"}:
            data['driveable'] = False
        elif highway == 'service':
            data['driveable'] = False
        elif highway == 'residential' and access == 'private':
            data['driveable'] = False

    ox.save_graphml(G, path)

    greeter.generate_spatial_layer()  # Set up spatial layer first
    greeter.creation_graph(options.file_name)  # Create the graph from .graphml file
    greeter.set_label()  # Set labels on nodes
    greeter.rename_junction_id()
    greeter.set_location()  # Set location attributes for nodes
    greeter.set_distance()  # Set distances between nodes/relationships
    greeter.rename_route_osmid()
    greeter.set_index()  # Set index for optimized queries
    greeter.close()  # Close the connection last

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
    return 0

main()
