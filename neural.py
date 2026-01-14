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





RES = 28 # 28x28 pixel canvas
PIX_MAX = 255 # pixel strength 0-255


train = pd.read_csv('MNIST/mnist_train.csv', index_col=0, header=None) # index (first col) = drawn number, header (first row) = pixel number 0-784 (28*28=784) -> 60000 x 784
train = train/PIX_MAX # set scale 0-1

test = pd.read_csv('MNIST/mnist_test.csv', index_col=0, header=None) # -""- -> 10000 x 784
test = test/PIX_MAX

data = test
SHAPE = data.shape


def makeWeightsBiases():
    
    """
    Initialization of weights & biases by random values
    """

    if not os.path.exists("WeightsBiases"):
        os.mkdir("WeightsBiases")


    RAN_SIZE = 10 # interval [-RAN_SIZE, +RAN_SIZE] for initialization of weights & biases
    LAYER_SIZE = 16 # number of layer neurons (both) Later change to see performance difference?
    OUT_SIZE = 10 # number of output neurons

    w1 = pd.DataFrame([[rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(SHAPE[1])] for j in range(LAYER_SIZE)]) # 16 x 784
    w1.to_csv("WeightsBiases/w1.csv", index=False, header=True)
    b1 = pd.DataFrame([rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)]) # 16
    b1.to_csv("WeightsBiases/b1.csv", index=False, header=True)
    w2 = pd.DataFrame([[rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)] for j in range(LAYER_SIZE)]) # 16 x 16
    w2.to_csv("WeightsBiases/w2.csv", index=False, header=True)
    b2 = pd.DataFrame([rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)]) # 16
    b2.to_csv("WeightsBiases/b2.csv", index=False, header=True)
    w3 = pd.DataFrame([[rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(LAYER_SIZE)] for j in range(OUT_SIZE)]) # 10 x 16
    w3.to_csv("WeightsBiases/w3.csv", index=False, header=True)
    b3 = pd.DataFrame([rn.uniform(-RAN_SIZE,RAN_SIZE) for i in range(OUT_SIZE)]) # 10
    b3.to_csv("WeightsBiases/b3.csv", index=False, header=True)





def ReLU(x): # Today ReLU(x) used instead of sigmoid(x)

    return max(0,x)

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def dsigmoid(x):

    return np.exp(-x) / (1 + np.exp(-x))**2





def activation(act, weight, bias):

    """
    Returns activation vector of next layer out of old activation vector a, weight matrix and bias vector
    """

    actNew = [None for i in range(len(bias))]

    for idx, (w, b) in enumerate(zip(weight, bias)):

        actNew[idx] = sigmoid(np.dot(act, w) + b[0])

    return actNew



def getActivations(inp, w1, b1, w2, b2, w3, b3):

    """
    Calculation for 1 run through network. Returns actual number in first element and all activations as lists
    """

    act_num = inp.name
    a_in = np.array(inp)

    a2 = activation(a_in, w1, b1)
    a3 = activation(a2, w2, b2)
    a_out = activation(a3, w3, b3)

    return act_num, a_in, a2, a3, a_out



def cost(act_num, neur_out):

    """
    Returns squared sum of neural output and the actual number as vector [0,0,..., 1,...] as measure of network performance for 1 single case
    """

    act_lst = [0 for j in range(len(neur_out))]
    act_lst[act_num] = 1

    zum = 0
    for i in range(len(neur_out)):
        zum += (neur_out[i] - act_lst[i])**2
   
    return zum

# Should cost function have all weights & biases as inputs? Above only really summed squared



def train():

    """
    Training the weights and biases with the complete dataset
    """

    w1 = np.array(pd.read_csv("WeightsBiases/w1.csv"))
    b1 = np.array(pd.read_csv("WeightsBiases/b1.csv"))
    w2 = np.array(pd.read_csv("WeightsBiases/w2.csv"))
    b2 = np.array(pd.read_csv("WeightsBiases/b2.csv"))
    w3 = np.array(pd.read_csv("WeightsBiases/w3.csv"))
    b3 = np.array(pd.read_csv("WeightsBiases/b3.csv"))

    n_out = len(w3)
    n3 = len(w2)
    n2 = len(w1)
    n_in = len(w1[0])

    cost_lst = list()
    for i in range(SHAPE[0]):

        print((i / SHAPE[0]) * 100, "%")

        act_num, a_in, a2, a3, a_out = getActivations(data.iloc[i], w1, b1, w2, b2, w3, b3)

        dw1, db1, dw2, db2, dw3, db3 = gradient(w1, b1, w2, b2, w3, b3, a_in, a2, a3, a_out, act_num)
        
        


        if i % 10 == 0 and i != 0:
            pass


        cost_lst.append(cost(act_num, a_out))


    avg_cost = np.mean(cost_lst)


    w1.to_csv("WeightsBiases/w1.csv", index=False, header=True)
    b1.to_csv("WeightsBiases/b1.csv", index=False, header=True)
    w2.to_csv("WeightsBiases/w2.csv", index=False, header=True)
    b2.to_csv("WeightsBiases/b2.csv", index=False, header=True)
    w3.to_csv("WeightsBiases/w3.csv", index=False, header=True)
    b3.to_csv("WeightsBiases/b3.csv", index=False, header=True)



def gradient(w1, b1, w2, b2, w3, b3, a_in, a2, a3, a_out, act_num):

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


    act_lst = [0 for j in range(n_out)]
    act_lst[act_num] = 1
    act_lst = np.array(act_lst)

    """
    dw3 = list()
    for i in range(n_out):
        jdx = list()
        for j in range(n3):
            jdx.append(2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i]) * a3[j])
        dw3.append(jdx)

    db3 = list()
    for i in range(n_out):
        db3.append(2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i]))
    """

    dw3, db3 = list(), list()
    for i in range(n_out):
        jdx = list()
        temp = 2 * (a_out[i] - act_lst[i]) * dsigmoid(z3[i])
        db3.append(temp)
        for j in range(n3): 
            jdx.append(temp * a3[j])
        dw3.append(jdx)
    
    """
    dw2 = list()
    for i in range(n3):
        temp = list()
        for j in range(n2):
            k_sum = sum([2 * (a_out[k] - act_lst[k]) * dsigmoid(z3[k]) * w3[k][i] for k in range(n_out)])
            temp.append(dsigmoid(z2[i]) * a2[j] * k_sum)
        dw2.append(temp)


    db2 = list()
    for i in range(n3):
        k_sum = sum([2 * (a_out[k] - act_lst[k]) * dsigmoid(z3[k]) * w3[k][i] for k in range(n_out)])
        db2.append(dsigmoid(z2[i]) * k_sum)
    """

    dw2, db2 = list(), list()
    for i in range(n3):
        jdx = list()
        k_sum = sum([2 * (a_out[k] - act_lst[k]) * dsigmoid(z3[k]) * w3[k][i] for k in range(n_out)])
        temp = dsigmoid(z2[i]) * k_sum
        db2.append(temp)
        for j in range(n2):
            jdx.append(temp * a2[j])
        dw2.append(jdx)

    """
    dw1 = list()
    for i in range(n2):
        jdx = list()
        for j in range(n_in):
            k_lst = list()
            for k in range(n_out):
                l_sum = sum([w3[k][l] * dsigmoid(z2[l]) * w2[l][i] for l in range(n3)])
                k_lst.append(2 * (a_out[k] - act_lst[k]) * dsigmoid(z3[k]) * l_sum)
            jdx.append(dsigmoid(z1[i]) * a_in[j] * sum(k_lst))
        dw1.append(jdx)

    db1 = list()
    for i in range(n2):
        k_lst = list()
        for k in range(n_out):
            l_sum = sum([w3[k][l] * dsigmoid(z2[l]) * w2[l][i] for l in range(n3)])
            k_lst.append( 2 * (a_out[k] - act_lst[k]) * dsigmoid(z3[k]) * l_sum)
        db1.append(dsigmoid(z1[i]) * sum(k_lst))
    """


    dw1, db1 = list(), list()
    for i in range(n2):
        jdx = list()
        k_lst = list()
        for k in range(n_out):
            l_sum = sum([w3[k][l] * dsigmoid(z2[l]) * w2[l][i] for l in range(n3)])
            k_lst.append(2 * (a_out[k] - act_lst[k]) * dsigmoid(z3[k]) * l_sum)
        temp = dsigmoid(z1[i]) * sum(k_lst)
        db1.append(temp)
        for j in range(n_in):
            jdx.append(temp * a_in[j])
        dw1.append(jdx)
    
    return dw3, db3, dw2, db2, dw1, db1


getData()

#makeWeightsBiases()

train()
