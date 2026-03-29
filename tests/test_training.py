import numpy as np
import random as rn
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neuralnetwork.training import sigmoid, dsigmoid, activation, cost


def test_sigmoid():

    assert sigmoid(0) == 0.5
    assert sigmoid(1) == 0.7310585786300049
    assert sigmoid(-1) == 0.2689414213699951
    assert sigmoid(np.inf) == 1
    assert sigmoid(-np.inf) == 0
    for x in [-100, -1, 0, 1, 100]:
        assert 0 <= sigmoid(x) <= 1

def test_dsigmoid():

    assert dsigmoid(0) == 0.25
    assert dsigmoid(1) == 0.19661193324148188
    assert dsigmoid(-1) == 0.19661193324148185
    assert dsigmoid(np.inf) == 0
    #assert dsigmoid(-np.inf) == 0
    for x in [-100, -1, 0, 1, 100]:
        assert 0 <= dsigmoid(x) <= 0.25

def test_activation():
    dim_in = rn.randint(1, 10)
    dim_out = rn.randint(1, 10)
    act = [np.random.rand() for _ in range(dim_in)]
    weight = np.random.randn(dim_out, dim_in)
    bias = [[np.random.randn()] for _ in range(dim_out)]
    res = [sigmoid((np.dot(weight, act) + bias[i])[i]) for i in range(dim_out)]
    res = [round(val, 15) for val in res]
    act_new = activation(act, weight, bias)
    act_new = [round(val, 15) for val in act_new]
    assert res == act_new
    for val in res:
        assert 0 <= val <= 1

def test_cost():
    act = [np.random.rand() for _ in range(10)]
    drawn_lst = [0 for j in range(10)]
    num = rn.randint(0, 9)
    drawn_lst[num] = 1
    cost_val = cost(num, act)
    assert cost_val == sum([(act[i] - drawn_lst[i]) ** 2 for i in range(10)])
    assert cost_val >= 0
    
    assert cost(9, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) == sum([(0.1 - 0) ** 2, (0.2 - 0) ** 2, (0.3 - 0) ** 2, (0.4 - 0) ** 2, (0.5 - 0) ** 2, (0.6 - 0) ** 2, (0.7 - 0) ** 2, (0.8 - 0) ** 2, (0.9 - 0) ** 2, (1 - 1) ** 2])
