# Neural network from scratch

This network will detect drawn numbers based on the MNIST dataset. In this code I will only implement basic Python functions and math (no nerual network/AI packages).


(Since training the weights and biases takes a couple of hours, an already trained set is provided.) 

<br>

The mathematics and understanding behind the code and this network in general are based on the neural network series by 3Blue1Brown on YouTube.

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

<br>

![image](network_image.jpg)

The MNIST dataset provides a 28 x 28 = 784 pixel grid which are used as input neurons. Each pixel/neuron represents how "painted" the pixel is from 0-1. The two hidden layer sizes $n_1 = n_2 = 16$ are arbitrary. The activation of the last layer $a^{(3)}$ represents how sure the network is about each number being the drawn one.

The activation $a^{(n)}$ of layer $n$ is calculated with the weight matrix $W^{(n)}$, its bias $b^{(n)}$ and the previous activation vector $a^{(n-1)}$ via the following equation.

$$a^{(n)} = W^{(n)} \cdot a^{(n-1)} + b^{(n)}, \qquad n = 1,2,3$$

Our goal is to minimize the cost function $C$, which is a measure on how well the network performs. The smaller the better.

$$C = \sum_{k=1}^{n_3} (a_k^{(3)} - y_k)^2$$

$y$ is a vector that represents the actual drawn number. For example if the number is $3$, $y = (0,0,0,1,0,0,...)^{T} $.

To minimize $C$ we need formulas for the gradient $\nabla C$, which are are listed below. ($\sigma$ represents the sigmoid function.)

$$\frac{\partial C}{\partial w_{ij}^{(3)}} = 2 (a_i^{(3)} - y_i ) \cdot \sigma^{\prime}(z_i^{(3)}) \cdot a_j^{(2)}$$

$$\frac{\partial C}{\partial b_{i}^{(3)}} = 2 (a_i^{(3)} - y_i ) \cdot \sigma^{\prime}(z_i^{(3)})$$


$$\frac{\partial C}{\partial w_{ij}^{(2)}} = \sigma^{\prime}(z_i^{(2)}) \cdot a_j^{(1)} \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot w_{ki}^{(3)}$$

$$\frac{\partial C}{\partial b_{i}^{(2)}} = \sigma^{\prime}(z_i^{(2)}) \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot w_{ki}^{(3)}$$


$$\frac{\partial C}{\partial w_{ij}^{(1)}} = \sigma^{\prime}(z_i^{(1)}) \cdot a_j^{(\text{in})} \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime}(z_l^{(2)}) \cdot w_{li}^{(2)}$$

$$\frac{\partial C}{\partial b_{i}^{(1)}} = \sigma^{\prime}(z_i^{(1)}) \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime}(z_l^{(2)}) \cdot w_{li}^{(2)}$$

$$\sigma(x) = \frac{1}{1+\text{e}^{-x}}$$
