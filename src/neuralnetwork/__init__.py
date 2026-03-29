"""
Public API for the neuralnetwork package.

Keeps package imports lightweight by exporting only core training/inference helpers.
"""

from .train import (
    getMNISTData,
    makeRandomWeightsBiases,
    getActivations,
    training,
    cost,
)

__all__ = [
    "getMNISTData",
    "makeRandomWeightsBiases",
    "getActivations",
    "training",
    "cost",
]