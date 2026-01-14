import numpy as np
import pandas as pd
import random as rn


def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def dsigmoid(x):

    return np.exp(-x) / (1 + np.exp(-x))**2


w1 = np.array([[-4,  6,  1,  7],
               [-3, -8, -3,  5],
               [ 2,  0, -1, -1]])
b1 = np.array([[ 1],
               [-3],
               [-7]])
w2 = np.array([[ 3, -4,  7],
               [-3,  3,  4],
               [10, -9,  2]])
b2 = np.array([[ -9],
               [  0],
               [ -1]])
w3 = np.array([[-10,  -7,   4],
               [  3,   0,  -1]])
b3 = np.array([[ 8],
               [-1]])

a_in = np.array([1, 0, 0.5, 0.79])


def activation(act, weight, bias):

    actNew = [None for i in range(len(bias))]

    for idx, (w, b) in enumerate(zip(weight, bias)):

        actNew[idx] = round(sigmoid(np.dot(act, w) + b[0]), 2)

    return actNew


def getActivations(inp, w1, b1, w2, b2, w3, b3):

    act_num = inp[0]
    a_in = inp[1]

    a2 = activation(a_in, w1, b1)
    a3 = activation(a2, w2, b2)
    a_out = activation(a3, w3, b3)

    return act_num, a_in, a2, a3, a_out


act_num, a_in, a2, a3, a_out = getActivations((1, a_in), w1, b1, w2, b2, w3, b3)

print("a_in")
print(act_num, a_in)
print()
print("a2")
print(a2)
print()
print("a3")
print(a3)
print()
print("a_out")
print(a_out)
print("\n\n")

n_out = len(a_out)
n3 = len(a3)
n2 = len(a2)
n_in = len(a_in)


z3, z2, z1 = list(), list(), list()
for i in range(n_out):
    z3.append(np.dot(a3, w3[i]) + b3[i][0])
for i in range(n3):
    z2.append(np.dot(a2, w2[i]) + b2[i][0])
for i in range(n2):
    z1.append(np.dot(a_in, w1[i]) + b1[i][0])

z1 = np.array(z1)
z2 = np.array(z2)
z3 = np.array(z3)

print("z1")
print(z1)
print()
print("z2")
print(z2)
print()
print("z3")
print(z3)
print()

act_lst = [0 for j in range(n_out)]
act_lst[act_num] = 1
act_lst = np.array(act_lst)

dw3 = list()
for i in range(n_out):
    temp = list()
    for j in range(n3):
        #temp.append(2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i]) * a3[j])
        
        dw3_ij = round( 2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i]) * a3[j] , 2)
        temp.append(dw3_ij)
    dw3.append(temp)

db3 = list()
for i in range(n_out):
    #db3.append(2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i]))

    db3_i = round( 2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i]) , 2)
    db3.append(db3_i)

print("dw3")
print(dw3)
print()
print("db3")
print(db3)
print()
