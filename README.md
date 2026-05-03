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

Reproducing Results

1. Install dependencies
For CNN-Attention-LSTM (PyTorch):
bashpip install torch numpy pandas scikit-learn matplotlib gdown

For LSTM (TensorFlow):
bashpip install tensorflow numpy pandas scikit-learn matplotlib gdown

2. Run the reproduce scripts

CNN-Attention-LSTM:
bashpython reproduce_results.py

Stacked LSTM:

bashpython reproduce_results_lstm.py

Checkpoints are downloaded automatically from Google Drive on first run. All figures are saved to ./figures/.

3. Run in Jupyter Notebook


!pip install torch tensorflow numpy pandas scikit-learn matplotlib gdown

CNN-Attention-LSTM
%run reproduce_results.py

LSTM
%run reproduce_results_lstm.py

References and results are included in the report submitted in gradescope.

For the three different scenarios of RUL and sequence length, use the following files.

reproduce_results_lstm_Copy1.py RUL = 110 seq_len = 25
reproduce_results_lstm_Copy2.py RUL = 100 seq_len = 20

