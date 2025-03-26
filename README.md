# bird_location_prediction

This repository contains the implementation and analysis of three recurrent neural network architectures (RNN, LSTM, GRU) for predicting bird flight trajectories.

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Implemented Models](#implemented-models)
- [Dataset](#dataset)
- [Results](#results)

## Project Overview

This project explores the application of deep learning techniques to predict bird flight trajectories, with the ultimate goal of reducing bird-turbine collisions in wind farms. The implementation compares three types of recurrent neural networks:

1. Basic Recurrent Neural Network (RNN)
2. Long Short-Term Memory (LSTM)
3. Gated Recurrent Unit (GRU)

The models are implemented in PyTorch and evaluated on real-world bird trajectory data.

## Motivation

Wind energy, while environmentally friendly, poses significant risks to avian populations:
- Estimated 500,000 bird deaths annually from turbine collisions in the US
- Potential increase to 1.4 million as wind farms expand
- Need for Automated Detection and Reaction Methods (ADaRM)

This project aims to develop accurate trajectory prediction models that can be integrated into collision avoidance systems, contributing to more sustainable wind energy solutions.

## Implemented Models

All models follow a sequence-to-sequence architecture with these common features:
- Configurable number of recurrent layers
- Adjustable hidden dimension size
- Multi-step (horizon) forecasting capability
- Batch-first processing
- Final linear projection layer

## Dataset

The models were trained on real bird trajectory data provided by wind farm company, containing:
- 3D coordinates (x, y, z)
- Timestamps
- Individual bird identifiers

### Preprocessing Steps:
1. Removal of implausible velocities (>10 m/s)
2. Sampling to 1-second intervals
3. Windowing (10-step input to predict 10-step output)
4. Train/validation/test split (80:10:10)

## Results

The models were evaluated across three experiments with increasing complexity:

### Experiment 1: Single Layer (hidden_dim=32)
| Model | Train RMSE | Val RMSE | Test RMSE |
|-------|-----------|----------|-----------|
| Naive | 12.18     | 11.92    | 12.19     |
| RNN   | 10.52     | 11.54    | 12.25     |
| GRU   | **8.60**  | **10.03**| **11.71** |
| LSTM  | 10.11     | 11.30    | 11.82     |

### Experiment 2: Single Layer (hidden_dim=128)
| Model | Train RMSE | Val RMSE | Test RMSE |
|-------|-----------|----------|-----------|
| Naive | 12.18     | 11.92    | 12.19     |
| RNN   | 5.21      | 6.71     | 8.77      |
| GRU   | **3.30**  | **5.00** | **5.49** |
| LSTM  | 3.45      | 6.04     | 6.85      |

### Experiment 3: Three Layers (hidden_dim=128)
| Model | Train RMSE | Val RMSE | Test RMSE |
|-------|-----------|----------|-----------|
| Naive | 12.18     | 11.92    | 12.19     |
| RNN   | 1.77      | 4.85     | 8.71      |
| GRU   | **0.85**  | **4.65** | **5.55** |
| LSTM  | 1.13      | 4.74     | 6.44      |

**Key Findings:**
- GRU consistently outperformed RNN and LSTM in generalization
- Increasing model complexity improved training performance but required regularization to prevent overfitting
- All neural models significantly outperformed the naive forecasting baseline
