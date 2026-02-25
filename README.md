# Traffic Speed Forecasting with Graph Neural Networks

## Overview

This project predicts future traffic speeds using a Graph Neural Network trained on the **PEMS-BAY** traffic dataset.

The model learns:
- How traffic changes over time  
- How connected road sensors influence each other  

It predicts traffic speeds **1 hour ahead** using past traffic data.

---

## Dataset

**PEMS-BAY** traffic dataset:

- Speed measurements from highway sensors  
- Collected every **5 minutes**  
- Each sensor is treated as a node in a graph  
- Sensors are connected using a road network adjacency matrix  

Files used:
- `pems-bay.h5` — traffic speed data  
- `adj_mx_bay.pkl` — road network connections  
- `pems-bay-meta.h5` — sensor location data  

---

## Problem Setup

- Input: past **12 time steps** (1 hour)
- Output: next **12 time steps** (1 hour ahead)
- Task: multi-step regression

The model predicts future speeds for all sensors at once.

---

## Model

A simplified **GraphWaveNet-style model** built in PyTorch.

It combines:
- Graph convolutions (to learn spatial relationships)
- Temporal layers (to learn time patterns)

Training uses:
- Adam optimizer  
- Evaluation metrics: MAE, RMSE, MAPE  

---

## Extra: Route Forecasting

The notebook also:

- Selects a path of connected sensors  
- Aggregates their predicted speeds  
- Plots predicted vs actual speeds  
- Visualizes prediction error  

This shows how the model can estimate traffic along a full route — not just individual sensors.

---

## Tech Stack

- Python  
- PyTorch  
- Pandas  
- NumPy  
- Matplotlib  
- NetworkX  
- (Optional) OSMnx for map visualization  

---

## How to Run

1. Place the dataset files in the working directory.
2. Open the notebook:

```bash
jupyter notebook
