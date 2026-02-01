import numpy as np
import matplotlib.pyplot as plt

with open("TrainedWBs/Trained_train/cost_train.txt", "r") as file:
    cost_lst = [float(line.strip()) for line in file]

plt.plot(cost_lst)
plt.ylabel("Cost")
plt.xlabel("Training cycles")
plt.grid()
plt.savefig("cost_plot.jpg")
plt.show()
