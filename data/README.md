# Data

This directory contains all network training and evaluation data, as well model parameter snapshots.

## Contents

- `MNIST/`: MNIST CSV datasets
- `models/`: Model snapshots (weights and biases as CSV files) with one longly trained model (`trained/`) and one that user can train directly (`current/`) in `training.py`, `gui.py`.

## Notes

- MNIST files are downloaded automatically when needed by `training.py`, `gui.py`.
