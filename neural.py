import numpy as np
import pandas as pd
import random as rn
import requests as req
import os
import shutil
import zipfile as zipf



def getData():
    """
    If the MNIST dataset is not on pc, this function will download and extract it
    """

    if not os.path.exists("MNIST"):
        os.mkdir("MNIST")

    if not os.path.isfile("MNIST/mnist_test.csv") or not os.path.isfile("MNIST/mnist_train.csv"):

        def downloadData(name):
            
            url = f"https://raw.githubusercontent.com/phoebetronic/mnist/main/mnist_{name}.csv.zip"

            response = req.get(url)
            response.raise_for_status()

            with open(f"MNIST/mnist_{name}.csv.zip", "wb") as file:
                file.write(response.content)

            with zipf.ZipFile(f"MNIST/mnist_{name}.csv.zip") as file:
                file.extractall("MNIST")

            os.remove(f"MNIST/mnist_{name}.csv.zip")


        downloadData("test")
        downloadData("train")

        shutil.rmtree("MNIST/__MACOSX")


getData()



RES = 28 # 28x28 pixel canvas
PIX_MAX = 255 # pixel strength 0-255
LAYER_SIZE = 16 # number of layer neurons
RAN_SIZE = 10 # interval [-RAN_SIZE, +RAN_SIZE] for initialization of weights & biases
OUT_SIZE = 10 # number of output neurons


train = pd.read_csv('MNIST/mnist_train.csv', index_col=0, header=None) # index (first col) = drawn number, header (first row) = inc number 0-784 -> 10000x784
train = train/PIX_MAX # set scale 0-1

test = pd.read_csv('MNIST/mnist_test.csv', index_col=0, header=None) # index (first col) = drawn number, header (first row) = inc number 0-784 -> 10000x784
test = test/PIX_MAX # set scale 0-1

data = test
SHAPE = data.shape # 10000 x 784


# Initialization of weights & biases
w1 = [[rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(SHAPE[1])] for j in range(LAYER_SIZE)]
b1 = [rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)]
w2 = [[rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)] for j in range(LAYER_SIZE)]
b2 = [rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)]
w3 = [[rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)] for j in range(OUT_SIZE)]
b3 = [rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(OUT_SIZE)]



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Today ReLU(x) used instead of sigmoid(x)
def ReLU(x):
    return max(0,x)

def activation(a, weight, bias):
    """Returns activation vector of next layer out of old activation vector a, weight matrix and bias vector"""
    a_new = [None for i in range(len(bias))]
    index = 0
    for w, b in zip(weight, bias):
        a_new[index] = sigmoid(np.dot(w, a) + b)
        index += 1
    return a_new



def output(input, w1, b1, w2, b2, w3, b3):
    """Calculation for 1 run through network. Returns actual number in first element and output activation list as second"""
    act_num = input.name
    act_inp = list(input)

    a1 = activation(act_inp, w1, b1)
    a2 = activation(a1, w2, b2)
    out = activation(a2, w3, b3)
    return act_num, out

otp = output(data.iloc[0], w1, b1, w2, b2, w3, b3)
print("n =", otp[0], [round(float(i),2) for i in otp[1]])



def cost(act_num, neur_out):
    """Returns squared sum of neural output and the actual number as vector [0,0,..., 1,...]"""
    act_num_lst = [0 for j in range(len(neur_out))]
    act_num_lst[act_num] = 1

    sum = 0
    for i in range(len(neur_out)):
        sum += (neur_out[i] - act_num_lst[i])**2
    return sum

print(cost(otp[0], otp[1]))



def avg_cost():
    # Average cost over all data as measure of network performance
    cost_lst = list()
    for i in range(SHAPE[0]):
        cost_lst.append(cost(data.iloc[i].name, output(data.iloc[i], w1, b1, w2, b2, w3, b3)[1]))
    return np.mean(cost_lst)

print(avg_cost())
