🌦️ AERIS

AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies

AERIS is an AI-driven extreme weather analysis and visualization platform designed to detect, track, analyze, and visualize severe weather anomalies using meteorological datasets, Graph Neural Networks (GNNs), spatio-temporal tracking, and deep-learning-based downscaling.

The system combines weather data processing, anomaly detection, spherical graph-based learning, trajectory tracking, conditional diffusion downscaling, and an interactive GIS dashboard into a unified software prototype.

> **AERIS is a research and demonstration prototype and is not an operational public weather-warning system.**

---

🎯 Problem Statement

Extreme weather events such as cyclones, storms, and other atmospheric anomalies are highly dynamic and spatially distributed.

Traditional weather analysis can involve large numerical datasets and complex spatial relationships, making it difficult to:

- Detect significant weather anomalies automatically
- Track the movement of weather systems over time
- Represent spatial relationships between weather-grid points
- Generate high-resolution spatial information from coarser forecast data
- Present complex meteorological information through an accessible interface

AERIS addresses these challenges through an integrated AI and geospatial processing pipeline.

---

 💡 Proposed Solution

AERIS processes meteorological data through multiple stages:

```text
Meteorological Data
        ↓
Data Ingestion & Validation
        ↓
Preprocessing & Quality Control
        ↓
Extreme Weather Anomaly Detection
        ↓
Spherical Graph Construction
        ↓
Graph Neural Network
        ↓
Spatio-Temporal Event Tracking
        ↓
Trajectory Generation
        ↓
Conditional Diffusion Downscaling
        ↓
GeoJSON / Analytical Outputs
        ↓
FastAPI Backend
        ↓
Next.js + Leaflet GIS Dashboard
