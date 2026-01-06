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


train = pd.read_csv('MNIST/mnist_train.csv', index_col=0, header=None) # index (first col) = drawn number, header (first row) = pixel number 0-784 (28*28=784) -> 60000 x 784
train = train/PIX_MAX # set scale 0-1

test = pd.read_csv('MNIST/mnist_test.csv', index_col=0, header=None) # -""- -> 10000 x 784
test = test/PIX_MAX

data = test
SHAPE = data.shape


def makeWeightsBiases():
    
    """
    Initialization of weights & biases
    """

    if not os.path.exists("WeightsBiases"):
        os.mkdir("WeightsBiases")


    RAN_SIZE = 10 # interval [-RAN_SIZE, +RAN_SIZE] for initialization of weights & biases
    LAYER_SIZE = 16 # number of layer neurons (both)
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



def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def ReLU(x): # Today ReLU(x) used instead of sigmoid(x)

    return max(0,x)



def oneRun(inp, w1, b1, w2, b2, w3, b3):

    """
    Calculation for 1 run through network. Returns actual number in first element and output activation list as second

    :return act_num: Actual drawn number
    :return out: Activation of last 10 neurons as list
    """

    def activation(act, weight, bias):

        """
        Returns activation vector of next layer out of old activation vector a, weight matrix and bias vector
        """

        actNew = [None for i in range(len(bias))]

        for idx, (w, b) in enumerate(zip(weight, bias)):

            actNew[idx] = sigmoid(np.dot(act, w) + b[0])

        return actNew


    act_num = inp.name
    a1_inp = np.array(inp)

    a2 = activation(a1_inp, w1, b1)
    a3 = activation(a2, w2, b2)
    out = activation(a3, w3, b3)

    return act_num, out



def cost(act_num, neur_out):

    """
    Returns squared sum of neural output and the actual number as vector [0,0,..., 1,...] as measure of network performance for 1 single case
    """

    act_num_lst = [0 for j in range(len(neur_out))]
    act_num_lst[act_num] = 1

    zum = 0
    for i in range(len(neur_out)):
        zum += (neur_out[i] - act_num_lst[i])**2
   
    return zum



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

    cost_lst = list()
    for i in range(SHAPE[0]):

        num, out = oneRun(data.iloc[i], w1, b1, w2, b2, w3, b3)

        w1, b1, w2, b2, w3, b3 = gradient(w1, b1, w2, b2, w3, b3)
        
        cost_lst.append(cost(num, out))


    avg_cost = np.mean(cost_lst)


    w1.to_csv("WeightsBiases/w1.csv", index=False, header=True)
    b1.to_csv("WeightsBiases/b1.csv", index=False, header=True)
    w2.to_csv("WeightsBiases/w2.csv", index=False, header=True)
    b2.to_csv("WeightsBiases/b2.csv", index=False, header=True)
    w3.to_csv("WeightsBiases/w3.csv", index=False, header=True)
    b3.to_csv("WeightsBiases/b3.csv", index=False, header=True)



def gradient(w1, b1, w2, b2, w3, b3):
    pass



#makeWeightsBiases()


"""
otp = oneRun(data.iloc[0], w1, b1, w2, b2, w3, b3)
print("n =", otp[0], [round(float(i),2) for i in otp[1]])
print(cost(otp[0], otp[1]))
"""