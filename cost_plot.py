import numpy as np
import matplotlib.pyplot as plt

with open("cost.txt", "r") as file:
    cost_lst = [float(line.strip()) for line in file]

plt.plot(cost_lst)
plt.ylabel("Cost")
plt.xlabel("Training cycle")
plt.show()
