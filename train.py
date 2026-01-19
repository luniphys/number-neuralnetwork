import numpy as np
import pandas as pd
import random as rn
import requests as req
import os
import shutil
import zipfile as zipf



def getMNISTData():

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





def makeRandomWeightsBiases():
    
    """
    Initialization of weights & biases by (small) random values
    """

    if not os.path.exists("WeightsBiases"):
        os.mkdir("WeightsBiases")


    LAYER_SIZE = 16 # number of layer neurons (both) Later change to see performance difference?
    OUT_SIZE = 10 # number of output neurons

    w1 = pd.DataFrame(np.random.randn(LAYER_SIZE, SHAPE[1]) * np.sqrt(1 / SHAPE[1])) # 16 x 784
    w1.to_csv("WeightsBiases/w1.csv", index=False, header=True)

    b1 = pd.DataFrame(np.random.randn(LAYER_SIZE) * np.sqrt(1 / LAYER_SIZE)) # 16
    b1.to_csv("WeightsBiases/b1.csv", index=False, header=True)

    w2 = pd.DataFrame(np.random.randn(LAYER_SIZE, LAYER_SIZE) * np.sqrt(1 / LAYER_SIZE)) # 16 x 16
    w2.to_csv("WeightsBiases/w2.csv", index=False, header=True)

    b2 = pd.DataFrame(np.random.randn(LAYER_SIZE) * np.sqrt(1 / LAYER_SIZE)) # 16
    b2.to_csv("WeightsBiases/b2.csv", index=False, header=True)

    w3 = pd.DataFrame(np.random.randn(OUT_SIZE, LAYER_SIZE) * np.sqrt(1 / LAYER_SIZE)) # 10 x 16
    w3.to_csv("WeightsBiases/w3.csv", index=False, header=True)

    b3 = pd.DataFrame(np.random.randn(OUT_SIZE) * np.sqrt(1 / OUT_SIZE)) # 10
    b3.to_csv("WeightsBiases/b3.csv", index=False, header=True)





def ReLU(x): # Today ReLU(x) used instead of sigmoid(x)

    return max(0,x)

def dReLU(x):

    if x <= 0:
        return 0
    
    else:
        return 1


def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def dsigmoid(x):

    return np.exp(-x) / (1 + np.exp(-x))**2





def activation(act, weight, bias):

    """
    Returns activation vector of next layer out of old activation vector act, weight matrix and bias vector
    """

    act_new = [None for i in range(len(bias))]

    for idx, (w, b) in enumerate(zip(weight, bias)):

        act_new[idx] = sigmoid(np.dot(act, w) + b[0])

    return act_new



def getActivations(inp, w1, b1, w2, b2, w3, b3):

    """
    Calculation for 1 run through network. Returns actual number in first element and all activations as lists
    """

    drawn_num = inp.name
    a_in = np.array(inp)

    a1 = activation(a_in, w1, b1)
    a2 = activation(a1, w2, b2)
    a3 = activation(a2, w3, b3)

    return drawn_num, a_in, a1, a2, a3



def cost(drawn_num, a3):

    """
    Returns squared sum of neural output and the actual drawn number as vector [0,0,..., 1,...] as measure of network performance for 1 single case
    """

    drawn_lst = [0 for j in range(len(a3))]
    drawn_lst[drawn_num] = 1

    squared_sum = 0
    for i in range(len(a3)):
        squared_sum += (a3[i] - drawn_lst[i])**2
   
    return squared_sum





def training():

    """
    Training the weights and biases with the complete dataset
    """

    w1 = np.array(pd.read_csv("WeightsBiases/w1.csv"))
    b1 = np.array(pd.read_csv("WeightsBiases/b1.csv"))
    w2 = np.array(pd.read_csv("WeightsBiases/w2.csv"))
    b2 = np.array(pd.read_csv("WeightsBiases/b2.csv"))
    w3 = np.array(pd.read_csv("WeightsBiases/w3.csv"))
    b3 = np.array(pd.read_csv("WeightsBiases/b3.csv"))

    n3 = len(w3)
    n2 = len(w2)
    n1 = len(w1)
    n_in = len(w1[0])

    LR = 0.1 # learning rate


    # Make dictionaries for each weight and bias as key and their gradients as items in lists for 10 cases
    dw3_dic, db3_dic = dict(), dict()
    for i in range(n3):
        db3_dic[f"db3_{i}"] = list()
        for j in range(n2):
            dw3_dic[f"dw3_{i},{j}"] = list()

    dw2_dic, db2_dic = dict(), dict()
    for i in range(n2):
        db2_dic[f"db2_{i}"] = list()
        for j in range(n1):
            dw2_dic[f"dw2_{i},{j}"] = list()

    dw1_dic, db1_dic = dict(), dict()
    for i in range(n1):
        db1_dic[f"db1_{i}"] = list()
        for j in range(n_in):
            dw1_dic[f"dw1_{i},{j}"] = list()
    

    cost_lst = list()
    for sample_idx in range(SHAPE[0]):

        #print(round((sample_idx / SHAPE[0]) * 100, 2), "%")

        drawn_num, a_in, a1, a2, a3 = getActivations(train.iloc[sample_idx], w1, b1, w2, b2, w3, b3)

        dw1, db1, dw2, db2, dw3, db3 = gradient(w1, b1, w2, b2, w3, b3, a_in, a1, a2, a3, drawn_num)


        for i in range(n3):
            db3_dic[f"db3_{i}"].append(-db3[i])
            for j in range(n2):
                dw3_dic[f"dw3_{i},{j}"].append(-dw3[i][j])

        for i in range(n2):
            db2_dic[f"db2_{i}"].append(-db2[i])
            for j in range(n1):
                dw2_dic[f"dw2_{i},{j}"].append(-dw2[i][j])

        for i in range(n1):
            db1_dic[f"db1_{i}"].append(-db1[i])
            for j in range(n_in):
                dw1_dic[f"dw1_{i},{j}"].append(-dw1[i][j])


        # Stochastic gradient descent. Use mean of 10 samples for time efficiency
        if sample_idx % 10 == 0 and sample_idx != 0 or sample_idx == SHAPE[0] - 1:

            for i in range(n3):
                b3[i] += LR * np.mean(db3_dic[f"db3_{i}"])
                db3_dic[f"db3_{i}"] = list()
                for j in range(n2):
                    w3[i][j] += LR * np.mean(dw3_dic[f"dw3_{i},{j}"])
                    dw3_dic[f"dw3_{i},{j}"] = list()

            for i in range(n2):
                b2[i] += LR * np.mean(db2_dic[f"db2_{i}"])
                db2_dic[f"db2_{i}"] = list()
                for j in range(n1):
                    w2[i][j] += LR * np.mean(dw2_dic[f"dw2_{i},{j}"])
                    dw2_dic[f"dw2_{i},{j}"] = list()

            for i in range(n1):
                b1[i] += LR * np.mean(db1_dic[f"db1_{i}"])
                db1_dic[f"db1_{i}"] = list()
                for j in range(n_in):
                    w1[i][j] += LR * np.mean(dw1_dic[f"dw1_{i},{j}"])
                    dw1_dic[f"dw1_{i},{j}"] = list()


        cost_lst.append(cost(drawn_num, a3))


    avg_cost = np.mean(cost_lst)
    with open("cost.txt", "a", encoding="utf-8") as file:
        file.write(f"{avg_cost}" + "\n")


    w1 = pd.DataFrame(w1)
    w1.to_csv("WeightsBiases/w1.csv", index=False, header=True)
    b1 = pd.DataFrame(b1)
    b1.to_csv("WeightsBiases/b1.csv", index=False, header=True)
    w2 = pd.DataFrame(w2)
    w2.to_csv("WeightsBiases/w2.csv", index=False, header=True)
    b2 = pd.DataFrame(b2)
    b2.to_csv("WeightsBiases/b2.csv", index=False, header=True)
    w3 = pd.DataFrame(w3)
    w3.to_csv("WeightsBiases/w3.csv", index=False, header=True)
    b3 = pd.DataFrame(b3)
    b3.to_csv("WeightsBiases/b3.csv", index=False, header=True)





def gradient(w1, b1, w2, b2, w3, b3, a_in, a1, a2, a3, drawn_num):

    n3 = len(a3)
    n2 = len(a2)
    n1 = len(a1)
    n_in = len(a_in)
    
    z3, z2, z1 = list(), list(), list()
    for i in range(n3):
        z3.append(np.dot(a2, w3[i]) + b3[i][0])
    for i in range(n2):
        z2.append(np.dot(a1, w2[i]) + b2[i][0])
    for i in range(n1):
        z1.append(np.dot(a_in, w1[i]) + b1[i][0])

    z1 = np.array(z1)
    z2 = np.array(z2)
    z3 = np.array(z3)


    drawn_lst = [0 for j in range(n3)]
    drawn_lst[drawn_num] = 1
    drawn_lst = np.array(drawn_lst) # = y


    dw3, db3 = list(), list()
    for i in range(n3):
        jdx = list()
        temp = 2 * (a3[i] - drawn_lst[i]) * dsigmoid(z3[i])
        db3.append(temp)
        for j in range(n2): 
            jdx.append(temp * a2[j])
        dw3.append(jdx)
    

    dw2, db2 = list(), list()
    for i in range(n2):
        jdx = list()
        k_sum = sum([2 * (a3[k] - drawn_lst[k]) * dsigmoid(z3[k]) * w3[k][i] for k in range(n3)])
        temp = dsigmoid(z2[i]) * k_sum
        db2.append(temp)
        for j in range(n1):
            jdx.append(temp * a1[j])
        dw2.append(jdx)


    dw1, db1 = list(), list()
    for i in range(n1):
        jdx = list()
        k_lst = list()
        for k in range(n3):
            l_sum = sum([w3[k][l] * dsigmoid(z2[l]) * w2[l][i] for l in range(n2)])
            k_lst.append(2 * (a3[k] - drawn_lst[k]) * dsigmoid(z3[k]) * l_sum)
        temp = dsigmoid(z1[i]) * sum(k_lst)
        db1.append(temp)
        for j in range(n_in):
            jdx.append(temp * a_in[j])
        dw1.append(jdx)
    
    return dw1, db1, dw2, db2, dw3, db3





if __name__ == "__main__":

    getMNISTData()

    PIX_MAX = 255 # pixel strength 0-255

    train = pd.read_csv('MNIST/mnist_train.csv', index_col=0, header=None) # index (first col) = drawn number, header (first row) = pixel number 0-784 (28*28=784) -> 60000 x 784
    train = train/PIX_MAX # set scale 0-1

    SHAPE = train.shape


    if not os.path.exists("WeightsBiases"):
        makeRandomWeightsBiases()


    cycles = 100
    for i in range(cycles):
        print(i)
        training()
