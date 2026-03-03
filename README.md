# Yale-UNC-CH-Geophysical-Waveform-Inversion


## Problem Overview

Seismic waveform inversion is a core problem in geophysics where the objective is to recover underground velocity structures from recorded seismic waves. In this competition, each sample contains a 3D seismic cube, and the task is to infer the corresponding 2D velocity model.

This is a supervised learning problem with pixel-wise regression on spatial grids.

## How to Run

### Train the Model

```python train.py --data_path <train_data> --epochs 100```

### Generate Submission

```python test.py --data_path <test_data> --model_path <saved_model>```