# DualRouteGNN

**Graph Attention for Traffic Volume Estimation and POI-Preference Routing from Sparse Sensors on Urban Road Networks**

---

## What is DualRouteGNN?

DualRouteGNN estimates traffic on road sections that have no sensor, and plans routes that pass through a Point of Interest (POI) the driver wants to visit on the way. A driver going from A to B can ask for a route that stops at a pharmacy and gets three options: the shortest, the low-traffic one, and a balanced one.

Only a small fraction of a city's roads carry a traffic sensor. The model learns from those roads and estimates Annual Average Daily Traffic (AADT) on the rest, using only what OpenStreetMap records about a road: no imagery, no probe vehicles, no speed profiles.

**Note on what AADT is.** AADT is an annual average daily volume. It carries no information about road capacity, delay, or time of day, so a low-traffic route is not for that reason a faster route. Travel time is reported separately, as free-flow time from length and speed limit.

---

## How It Works

| Step | What happens |
|------|-------------|
| 1. Graph Construction | Road network and POIs are downloaded from OpenStreetMap and stored in Neo4j |
| 2. Feature Engineering | A dual graph is built where each node is one OSM way (a *road section*). Each section gets geometry (length, lanes, speed limit), road type, centrality (PageRank, betweenness, degree) and POI counts by category, giving 26 features in Madrid and 25 in Paris |
| 3. GAT Model | A Graph Attention Network runs on the whole dual graph while the loss is computed only on the measured sections, so unmeasured sections still take part in message passing |
| 4. Routing | A* on the primal graph with cost `d/100 × (α_d + α_t · AADT/T_ref)`. The POI stop is chosen under the same cost, not by distance alone |

### Three levels of the road network

A named **street** is made of one or more OSM **ways**. A way is stored as a chain of **segments**, and a segment joins two consecutive junctions and forms one edge of the primal graph. Each node of the dual graph is one way, and the code calls it a **road section**. A sensor is recorded on a way, so it labels one road section rather than a whole street.

---

## Evaluation

Accuracy on a random split is misleading here, because a test road usually sits next to a road that was in training. Every result is therefore reported under four ways of forming the folds, and each one carries a **neighbour-leak rate**: the share of test sections directly connected in the graph to a training section.

| Protocol | How folds are formed |
|----------|---------------------|
| Random | 5-fold random split |
| Spatial block | 2 km grid cells, a whole cell goes to one fold |
| District | 5 contiguous regions from KMeans on coordinates, held out in turn |
| Corridor | GroupKFold on street name, so every section of a street stays together |

Every measured road section is estimated exactly once, while it was held out, and the metrics are computed over all of them together. Nothing is selected on the test data: training runs for a fixed number of epochs and the final parameters are the ones reported.

---

## Results

Under the **district** protocol, the strictest one, with pooled out-of-fold estimates:

| | Madrid | Paris |
|---|---|---|
| Road sections | 58,854 | 44,197 |
| Sections with a sensor | 1,133 (1.93%) | 1,256 (2.84%) |
| Median AADT | 2,996 | 9,124 |
| **R²** | **0.701** | **0.949** |
| MAE | 2,030 | 2,245 |
| RMSE | 3,640 | 4,340 |
| SMAPE | 48.5% | 19.1% |
| Class-mean baseline | 0.587 | 0.852 |
| Best other baseline | Gradient Boosting 0.644 | MLP 0.868 |
| R² spread over the four protocols | 0.026 | 0.026 |

**The two cities are not equally difficult.** Paris sensors sit mostly on the major network, so road class alone already explains much of the variation there and every model scores higher. Madrid sensors are spread across road classes and the task is harder. What carries across both cities is the behaviour under the protocols: the model's R² moves by the same 0.026 between the easiest and the hardest split, while a GCN on the same skeleton moves by 0.24 in Madrid and 0.11 in Paris.

### Baselines

Thirteen baselines, all trained and tested on the same folds, in four groups:

| Group | Models |
|-------|--------|
| Road attributes only | Linear Regression, Random Forest, Gradient Boosting, MLP, SVR, GPR |
| Position only | Nearest-sensor, IDW, Kriging |
| Road class | Class mean, IDW within the same class |
| Attributes and network | GCN, GraphSAGE |

GCN and GraphSAGE use the same skeleton as the GAT, with the same widths, skip connections, normalisation and predictor head, so the only difference between them is the aggregation operator.

### Routing (Madrid)

| Option | Traffic | Distance | Free-flow time |
|--------|---------|----------|----------------|
| Low-traffic vs shortest | −53% to −58% | +27% to +36% | +40% to +58% |
| At α_t = 0.25 | −43% | +9% | n/a |

Most of the benefit arrives early: a quarter weight on traffic already avoids 43% of the traffic for 9% more distance. POI types tested: clinic, pharmacy, fuel, restaurant, school.

---

## Getting Started

### Requirements

- Python 3.10+
- Neo4j 5.26.x with the Graph Data Science plugin (GDS 2.x) and APOC
- PyTorch and PyTorch Geometric
- OSMnx, Folium, Pandas, NumPy, scikit-learn, SciPy, PyKrige

```bash
pip install -r requirements.txt
```

### Step 1: Build the Road Graph

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
| `-x` | Latitude of the city centre |
| `-y` | Longitude of the city centre |
| `-d` | Radius in metres |
| `-n` | Neo4j connection URI |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-f` | Output GraphML filename |

For Paris, use `-x 48.8566 -y 2.3522 -d 7000 -f paris.graphml`.

### Step 2: Download and Connect POIs

```bash
python step1_graph_construction/add_pois.py \
    -x 40.4168 \
    -y -3.7038 \
    -d 11000 \
    -n neo4j://localhost:7687 \
    -u neo4j \
    -p your_password
```

### Step 3: Attach the Measured AADT

```bash
python step1_graph_construction/add_aadt.py \
    -n neo4j://localhost:7687 \
    -u neo4j \
    -p your_password \
    --csv Madrid_AADT_clean.csv
```

### Step 4: Feature Engineering

Compute centrality, extract features, and build the dual graph:

```bash
python step2_feature_engineering/extract_features.py
```

### Step 5: Train the GAT and Estimate Traffic

Runs the four protocols, collects pooled out-of-fold estimates, then trains a final model on every measured section and writes an AADT estimate for the whole network:

```bash
python step3_gat_training/gat.py
```

Outputs: `oof_predictions_gat.csv`, `pooled_metrics_gat.csv`, `leakage_audit.csv`, `aadt_estimates.csv`, `model_final.pt`.

### Step 6: Compare Against the Baselines

Runs the thirteen baselines on the same folds and compares each one against the GAT road section by road section, with bootstrap intervals on the R² gap:

```bash
python step3_gat_training/baselines.py
```

Run this **after** Step 5, because it reads `oof_predictions_gat.csv` rather than retraining the GAT.

### Step 7: Run the Routing

```bash
python step4_routing/route_compute.py
python step4_routing/route_visualize.py
python step4_routing/route_batch_evaluation.py
```

`route_setup.py` loads the network and defines the cost function, A*, and the stop selection; the other three import from it. `route_batch_evaluation.py` also produces the weight sweep, the comparison between the joint stop choice and choosing the stop by distance alone, and the sensitivity of the routes to the estimation error.

---

## Cities

| | Madrid, Spain | Paris, France |
|---|---|---|
| Radius from centre | 11 km | 7 km |
| Road sections | 58,854 | 44,197 |
| Sections with a sensor | 1,133 (1.93%) | 1,256 (2.84%) |

Traffic data from: Bonnemaizon et al., *Harmonized Annual Averaged Traffic Data at Street Segment Level for European Cities*, Scientific Data, 2025.

---

## What the Method Does Not Do

- Accuracy on unmeasured road sections cannot be evaluated, because no measurement exists for them.
- The model does not extrapolate to a road class that is absent from the sensors. The estimates apply to unmeasured sections of classes that are represented among the sensors.
- No observed travel-time or speed data was available for either city, so the routes were not checked against real journeys. The reported time is free-flow time and contains no delay.
