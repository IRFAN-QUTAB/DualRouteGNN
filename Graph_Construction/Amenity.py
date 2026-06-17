import overpy
import json
from neo4j import GraphDatabase
import argparse
import os
import time


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

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

    def import_node(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_node)
            return result

    @staticmethod
    def _import_node(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
            "CALL apoc.load.json('nodefile.json') 
            YIELD value
            UNWIND value.elements AS nodo
            return nodo",
            "MERGE (n:OSMNode:POI {osm_id: nodo.id})
            ON CREATE SET n.name=nodo.tags.name,
            n.lat=tofloat(nodo.lat), 
            n.lon=tofloat(nodo.lon), 
            n.geometry= 'POINT(' + nodo.lat + ' ' + nodo.lon +')'
            WITH n, nodo
            MERGE (n)-[:TAGS]->(t:Tag)
            ON CREATE SET t += nodo.tags", 
            {batchSize:100, iterateList:true, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total;
        """)
        return result.values()

    def import_node_way(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_node_way)
            return result

    @staticmethod
    def _import_node_way(tx):
        result = tx.run("""
                CALL apoc.periodic.iterate(
                "CALL apoc.load.json('nodeway.json') 
                YIELD value
                UNWIND value.elements AS node
                return node",
                "MERGE (n:OSMNode {osm_id: node.id})
                ON CREATE SET 
                n.lat = toFloat(node.lat),
                n.lon = toFloat(node.lon),
                n.geometry = 'POINT(' + node.lat + ' ' + node.lon +')'", 
                {batchSize:100, iterateList:true, parallel:false}
                )
                YIELD batches, total
                RETURN batches, total;
            """)
        return result.values()

    def import_way(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_way)
            return result

    @staticmethod
    def _import_way(tx):
        result = tx.run("""
                CALL apoc.periodic.iterate(
                "CALL apoc.load.json('wayfile.json') 
                YIELD value
                UNWIND value.elements AS el
                return el",
                "MERGE (w:OSMWay:POI {osm_id: el.id}) 
                ON CREATE SET w.name = el.tags.name
                MERGE (w)-[:TAGS]->(t:Tag) 
                ON CREATE SET t += el.tags
                WITH w, el.nodes AS nodes
                UNWIND nodes AS node
                MATCH (n:OSMNode {osm_id: node})
                MERGE (n)-[:PART_OF]->(w)", 
                {batchSize:100, iterateList:true, parallel:false}
                )
                YIELD batches, total
                RETURN batches, total;
            """)
        return result.values()

    def import_nodes_into_spatial_layer(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_nodes_into_spatial_layer)
            return result

    @staticmethod
    def _import_nodes_into_spatial_layer(tx):
        result = tx.run("""
            MATCH (n:OSMNode)
            CALL spatial.addNode('spatial', n) yield node return node;
        """)
        return result.values()

    def set_location(self):
        """Insert the location in the POI, OSMNode, and RoadJunction nodes."""
        with self.driver.session() as session:
            result = session.execute_write(self._set_location)
            return result

    @staticmethod
    def _set_location(tx):
        result = tx.run("""
                MATCH (n:OSMNode) 
                SET n.location = point({latitude: tofloat(n.lat), longitude: tofloat(n.lon)})
            """)
        return result.values()

    def set_index(self):
        with self.driver.session() as session:
            try:
                result = session.execute_write(self._set_index)
                return result
            except Exception as e:
                print(f"Index creation skipped or failed: {e}")

    @staticmethod
    def _set_index(tx):
        try:
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:OSMNode) ON (n.osm_id)")
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:PointOfInterest) ON (n.osm_id)")
        except Exception as e:
            print(f"Failed to create index: {e}")
        return []

    def mark_driveable_roadjunctions(self):
        with self.driver.session() as session:
            session.execute_write(self._mark_driveable_roadjunctions)

    @staticmethod
    def _mark_driveable_roadjunctions(tx):
        tx.run("""
               MATCH ()-[r:ROUTE]-()
               WHERE r.driveable = 'True' OR r.driveable = 'true'
               SET r.driveable = true
           """).consume()

        tx.run("""
               MATCH (n:RoadJunction)
               SET n.driveable = false
           """).consume()

        tx.run("""
               MATCH (n:RoadJunction)-[r:ROUTE]-()
               WHERE r.driveable = true
               SET n.driveable = true
           """).consume()

    def connect_amenity(self):
        """Connect the POI and OSMNode to the nearest RoadJunction."""
        with self.driver.session() as session:
            result = session.execute_write(self._connect_amenity)

    @staticmethod
    def _connect_amenity(tx):
        result = tx.run("""
            MATCH (osmn:OSMNode)
            WHERE NOT (osmn)-[:NEAR]->(:RoadJunction)
              AND exists(osmn.lat) AND exists(osmn.lon)
            WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS nodeLoc
            MATCH (rj:RoadJunction)
            WHERE rj.driveable = true AND exists(rj.location) AND distance(rj.location, nodeLoc) < 100
            WITH osmn, rj, distance(rj.location, nodeLoc) AS dist
            ORDER BY dist
            WITH osmn, collect({rj: rj, dist: dist})[0] AS nearest
            WHERE nearest IS NOT NULL
            WITH osmn, nearest.rj AS nearestRJ, nearest.dist AS nearestDist
            MERGE (osmn)-[r:NEAR]->(nearestRJ)
              ON CREATE SET r.distance = nearestDist, r.status = 'driveable_within_100m'
        """)

        result = tx.run("""
            MATCH (osmn:OSMNode)
            WHERE NOT (osmn)-[:NEAR]->(:RoadJunction)
              AND exists(osmn.lat) AND exists(osmn.lon)
            WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
            CALL {
              WITH osmn, osmLoc
              MATCH (rj:RoadJunction)
              WHERE rj.driveable = false AND exists(rj.location)
              RETURN rj, distance(osmLoc, rj.location) AS dist
              ORDER BY dist ASC
              LIMIT 1
            }
            MERGE (osmn)-[r:NEAR]->(rj)
              ON CREATE SET r.distance = dist, r.status = 'nearest_non_driveable'
        """)

        result = tx.run("""
            MATCH (poi:OSMWay:POI)<-[:PART_OF]-(osmn:OSMNode)
            WHERE NOT (osmn)-[:NEAR]->(:RoadJunction {driveable: true})
              AND (osmn)-[:NEAR]->(:RoadJunction {driveable: false})
            WITH poi, collect(osmn) AS nodes
            WHERE size(nodes) > 0
            WITH poi, nodes[0] AS chosenNode
            WITH chosenNode,
                 point({latitude: toFloat(chosenNode.lat), longitude: toFloat(chosenNode.lon)}) AS osmLoc
            MATCH (drvRJ:RoadJunction)
            WHERE drvRJ.driveable = true AND exists(drvRJ.location)
            WITH chosenNode, drvRJ,
                 distance(osmLoc, drvRJ.location) AS dist
            ORDER BY dist ASC
            LIMIT 1
            MERGE (chosenNode)-[r:NEAR]->(drvRJ)
              ON CREATE SET
                r.distance = dist,
                r.status = 'nearest_driveable_from_non_driveable_poi'
        """)

        tx.run("""
                            MATCH (p:POI)-[r:PART_OF]->(p2:POI)
                            DELETE r
                        """).consume()

        return result.values()


def add_options():
    parser = argparse.ArgumentParser(description='Insertion of POI in the graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--spatial', '-s', dest='spatial', type=str, required=False, default='False')
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)
    api = overpy.Overpass(url="https://overpass.kumi.systems/api/interpreter")
    .
    dist = options.dist
    lon = options.lon
    lat = options.lat
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    path = greeter.get_path()[0][0] + '\\' + greeter.get_import_folder_name()[0][0] + "\\"

    # Original around query — no changes needed
    result = api.query(f"""
            [out:json][timeout:120];
            (
                way(around:{dist},{lat},{lon})["amenity"];
            );(._;>;);
            out body;
        """)

    # generate json file with nodes that compose each way
    list_node_way = []
    for w in result.ways:
        print(w)
        for n in w.get_nodes(resolve_missing=False):
            d = {'type': 'node', 'id': n.id,
                 'id_way': w.id,
                 'lat': str(n.lat),
                 'lon': str(n.lon),
                 'geometry': 'POINT(' + str(n.lat) + ' ' + str(n.lon) + ')',
                 'tags': n.tags}
            print(d)
            list_node_way.append(d)
    res = {"elements": list_node_way}
    print("nodes to import:")
    print(res)
    print("-----------------------------------------------------------------------")
    with open(path + 'nodeway.json', "w") as f:
        json.dump(res, f)
        print("file generated in import directory")

    greeter.import_node_way()

    list_way = []
    for way in result.ways:
        d = {'type': 'way', 'id': way.id, 'tags': way.tags}
        l_node = []
        for node in way.nodes:
            l_node.append(node.id)
        d['nodes'] = l_node
        list_way.append(d)
    res = {"elements": list_way}
    print("ways to import:")
    print(res)
    print("-----------------------------------------------------------------------")
    with open(path + "wayfile.json", "w") as f:
        json.dump(res, f)
        print("file generated in import directory")

    greeter.import_way()
    print("import wayfile.json: done")

    # query overpass API for POI represented as nodes
    result = api.query(f"""
    [out:json][timeout:120];
    (
      node["amenity"](around:{dist},{lat},{lon});
      way["amenity"](around:{dist},{lat},{lon});
    );
    out center;
    """)

    list_node = []
    for node in result.nodes:
        d = {'type': 'node', 'id': node.id,
             'lat': str(node.lat),
             'lon': str(node.lon),
             'geometry': 'POINT(' + str(node.lat) + ' ' + str(node.lon) + ')',
             'tags': node.tags}
        list_node.append(d)
    res = {"elements": list_node}
    print("nodes to import:")
    print(res)
    print("-----------------------------------------------------------------------")
    with open(path + 'nodefile.json', "w") as f:
        json.dump(res, f)
    print("file generated in import directory")

    greeter.import_way()
    greeter.import_node()

    if options.spatial == 'True':
        greeter.import_nodes_into_spatial_layer()

    greeter.set_location()

    greeter.mark_driveable_roadjunctions()
    greeter.connect_amenity()
    greeter.close()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    return 0


main()
