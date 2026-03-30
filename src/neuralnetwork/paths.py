import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] # number-neuralnetwork/

DATA_DIR = BASE / "data/"

MNIST_DIR = DATA_DIR / "MNIST/"

MODELS_DIR = DATA_DIR / "models/"
CURRENT_DIR = MODELS_DIR / "current/"
TRAINED_DIR = MODELS_DIR / "trained/training/"

SRC_DIR = BASE / "src/"
NEURALNETWORK_DIR = SRC_DIR / "neuralnetwork/"
ASSETS_DIR = NEURALNETWORK_DIR / "assets/"
