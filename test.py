import numpy as np
import pandas as pd
import random as rn
import time

from train import getActivations, cost



def average_cost():

    """
    Prints the average cost of the network for the whole test data. Can be seen as network error propability.
    """

    cost_lst = list()
    for sample_idx in range(SHAPE[0]):

        drawn_num, a3 = getActivations(test.iloc[sample_idx], w1, b1, w2, b2, w3, b3)[::4]

        cost_lst.append(cost(drawn_num, a3))

    avg_cost = np.mean(cost_lst)
    print("Average Cost", round(avg_cost, 5))



def get_guess(sample):

    """
    Returing the networks guessed number from an input sample 
    
    :param a3: Description
    """

    a3 = getActivations(sample, w1, b1, w2, b2, w3, b3)[4]

    return max(enumerate(a3), key = lambda x: x[1])[0]


def try_random_num():

    """
    Get a random sample out of the test dataset and print the networks guess.
    """

    ran_sample = test.iloc[rn.randint(0, SHAPE[0])]
    drawn_num = ran_sample.name

    print("The drawn number is", drawn_num, "!")
    time.sleep(0.65)
    print("Thinking ", end="")
    for i in range(3):
        print(". ", end="", flush=True)
        time.sleep(0.65)

    guessed_num = get_guess(ran_sample)

    print("\nThe number should be", end=" ")
    print(guessed_num, "!", end=" ")

    if drawn_num == guessed_num:
        print("Correct! :)")

    else:
        print("Wrong :(")



PIX_MAX = 255 # pixel strength 0-255

test = pd.read_csv('MNIST/mnist_test.csv', index_col=0, header=None) # -""- -> 10000 x 784
test = test/PIX_MAX

SHAPE = test.shape


w1 = np.array(pd.read_csv("WeightsBiases/w1.csv"))
b1 = np.array(pd.read_csv("WeightsBiases/b1.csv"))
w2 = np.array(pd.read_csv("WeightsBiases/w2.csv"))
b2 = np.array(pd.read_csv("WeightsBiases/b2.csv"))
w3 = np.array(pd.read_csv("WeightsBiases/w3.csv"))
b3 = np.array(pd.read_csv("WeightsBiases/b3.csv"))


average_cost()

try_random_num()
