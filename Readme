# DualRouteGNN

**Graph Attention for Traffic Volume Estimation and POI-Preference Routing from Sparse Sensors on Urban Road Networks**

## Overview

DualRouteGNN is a framework that estimates Annual Average Daily Traffic (AADT) on roads without sensors and routes drivers through user-chosen Points of Interest (POIs). The system builds a segment-level dual graph from OpenStreetMap data in Neo4j, trains a Graph Attention Network on sparse real-world sensor data, and computes multi-objective routes through a selected POI type.

## Pipeline

1. **Data Source** — Road network from OpenStreetMap (via OSMnx), POIs from Overpass API, real AADT from the European harmonized dataset.
2. **Graph Construction** — Primal graph (junctions and roads) stored in Neo4j, transformed into a segment-level dual graph (58,854 nodes).
3. **Feature Engineering** — 26 features per road: geometry, centrality, POI counts, road type.
4. **GAT Model** — Three-layer Graph Attention Network estimates AADT, trained on <2% of roads with real counts.
5. **Multi-objective Routing** — Routes through a user-chosen POI with three options: shortest, least-traffic, and balanced.

## Results

- GAT achieves R² = 0.74 on test roads, outperforming LR, RF, MLP, and GCN baselines.
- Across five POI types (clinic, pharmacy, fuel, restaurant, school), the least-traffic route reduces traffic exposure by 62–66% relative to the shortest route.

## City

- **Madrid, Spain** — 58,854 road segments, 1,133 with real AADT sensors (1.93% coverage).
- Data from: Bonnemaizon et al., "Harmonized Annual Averaged Traffic Data at Street Segment Level for European Cities," Scientific Data, 2025.

## Requirements

- Python 3.10+
- Neo4j 4.4.x with GDS plugin
- PyTorch, PyTorch Geometric
- OSMnx, Folium, Pandas, NumPy

## Repository Structure
