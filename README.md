# Traffic Speed Forecasting with Graph Neural Networks

## Overview

This project predicts future traffic speeds using a Graph Neural Network trained on the **PEMS-BAY** traffic dataset.

The model learns:

* Temporal patterns in traffic flow
* Spatial relationships between road sensors via a graph structure

In addition to forecasting, the project visualizes how predictions relate to the underlying road network and along a selected route.

---

## Dataset

**PEMS-BAY** traffic dataset:

* Speed measurements from highway sensors
* Collected every **5 minutes**
* Each sensor is represented as a node in a graph
* Connections between sensors are defined by an adjacency matrix based on road proximity

Files used:

* `pems-bay.h5` — traffic speed data
* `adj_mx_bay.pkl` — adjacency matrix (sensor connectivity + weights)
* `pems-bay-meta.h5` — sensor metadata

---

## Problem Setup

* Input: past **12 time steps** (1 hour)
* Output: next **12 time steps** (1 hour ahead)
* Task: multi-step time-series regression

The model predicts future speeds for all sensors simultaneously.

---

## Model

A simplified **GraphWaveNet-style model** implemented in PyTorch.

It combines:

* Graph-based layers to model spatial dependencies between sensors
* Temporal structure to capture short-term traffic dynamics

Training setup:

* Optimizer: Adam
* Metrics: MAE, RMSE, MAPE

---

## Graph Representation

The road network is represented as a directed graph:

* Nodes = traffic sensors
* Edges = road connections between sensors
* Edge weights = proximity-based strength of connection

A local subgraph is extracted around a selected sensor (2-hop neighborhood) to visualize the structure of the network. Edge color intensity reflects connection strength.

---

## Route-Level Forecasting and Visualization

Beyond aggregate metrics, the notebook evaluates predictions along a specific route:

* A sequence of connected sensors is selected
* Model predictions are extracted along this route
* Predicted vs. actual speeds are plotted over time
* Prediction error is visualized per timestep

Additionally, predictions are projected onto the route graph, providing a spatial view of traffic intensity across segments.

These visualizations help identify behaviors such as smoothing and uneven error across different parts of the route.

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* NetworkX

---

## How to Run

1. Place the dataset files in the working directory
2. Open the notebook:

```bash
jupyter notebook
```

