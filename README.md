# DualRouteGNN

**Graph Attention for Traffic Volume Estimation and POI-Preference Routing from Sparse Sensors on Urban Road Networks**

---

## What is DualRouteGNN?

DualRouteGNN estimates traffic on roads that have no sensors and finds routes that pass through a Point of Interest (POI) the driver wants to visit on the way. For example, a driver going from A to B can ask the system to find a route that passes through a pharmacy, and get three options: the shortest, the one with the least traffic, and a balanced one.

The system uses real traffic sensor data from a small fraction of roads to learn traffic patterns and estimate traffic on the rest.

---

## How It Works

| Step | What happens |
|------|-------------|
| 1. Graph Construction | Road network and POIs are downloaded from OpenStreetMap and stored in Neo4j |
| 2. Feature Engineering | Each road gets 26 features: length, lanes, speed, centrality, POI counts, road type |
| 3. GAT Model | A Graph Attention Network learns from roads with sensors and estimates traffic on the rest |
| 4. Routing | The system finds a road with the chosen POI and computes three routes through it |

---

## Results

| Metric | Value |
|--------|-------|
| R² on test roads | 0.74 |
| Baselines beaten | Linear Regression, Random Forest, MLP, GCN |
| Traffic reduction (least-traffic vs shortest) | 62–66% |
| POI types tested | clinic, pharmacy, fuel, restaurant, school |

---

## Getting Started

### Requirements

- Python 3.10+
- Neo4j 4.4.x with Graph Data Science plugin
- PyTorch and PyTorch Geometric
- OSMnx, Folium, Pandas, NumPy

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Build the Road Graph

Download the road network from OpenStreetMap and load it into Neo4j:

```bash
python step1_graph_construction/build_graph.py \
    -x 40.4168 \
    -y -3.7038 \
    -d 11000 \
    -n neo4j://localhost:7687 \
    -u neo4j \
    -p your_password \
    -f madrid.graphml
```

| Parameter | Description |
|-----------|-------------|
| `-x` | Latitude of the city center |
| `-y` | Longitude of the city center |
| `-d` | Radius in meters |
| `-n` | Neo4j connection URI |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-f` | Output GraphML filename |

### Step 2: Download and Connect POIs

Download Points of Interest from OpenStreetMap and connect them to the road network:

```bash
python step1_graph_construction/add_pois.py \
    -x 40.4168 \
    -y -3.7038 \
    -d 11000 \
    -n neo4j://localhost:7687 \
    -u neo4j \
    -p your_password
```

### Step 3: Feature Engineering

Compute centrality measures, extract features, and build the dual graph:

```bash
python step2_feature_engineering/extract_features.py
```

### Step 4: Train the GAT Model

Train the model on roads with real AADT and estimate traffic on the rest:

```bash
python step3_gat_training/train.py
```

### Step 5: Run Routing

Compute routes through a chosen POI:

```bash
python step4_routing/route_compute.py
```

Visualize routes on a map:

```bash
python step4_routing/route_visualize.py
```

Run batch evaluation across 5 POI types:

```bash
python step4_routing/route_batch_evaluation.py
```

---

## City

- **Madrid, Spain**
- 58,854 road segments, 1,133 with real AADT sensors (1.93% coverage)
- Traffic data from: Bonnemaizon et al., *Harmonized Annual Averaged Traffic Data at Street Segment Level for European Cities*, Scientific Data, 2025.

---
