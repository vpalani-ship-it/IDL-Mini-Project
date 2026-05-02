# RUL Prediction using CNN–Attention–LSTM (C-MAPSS)

## Overview
This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA **C-MAPSS dataset**, a benchmark problem in predictive maintenance.

We compare:
- **Baseline:** LSTM model  
- **Variant:** CNN–Attention–LSTM model  

The goal is to evaluate how architectural changes (CNN + Attention) improve performance over standard sequence modeling.

---

## Dataset

- Dataset: NASA C-MAPSS (FD001 subset)
- Source: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

### Description
- Multivariate time-series data
- 21 sensor measurements + 3 operational settings
- Task: Predict RUL (number of cycles before failure)

