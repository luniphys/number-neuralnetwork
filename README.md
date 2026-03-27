# Neural network from scratch

This network will detect drawn numbers based on the <b>MNIST</b> dataset. In this code I only implemented <b>basic</b> Python functionality and math (no neural network/AI packages).

### Training & Testing

The complete backend logic is manifested in <i>train.py</i>, where the network is set up and all the weights and biases are trained with the large <b>MNIST</b> training dataset (60.000 datapoints) by a self implemented backpropagation.

In <i>test.py</i>, I test the network with the <b>MNIST</b> test dataset (10.000 datapoints). Overall I get a <b>94.84%</b> accuracy after training the network for ~ 60 hours in 281 training cycles. Below you can see the <b><i>cost value</i></b> trend during training.

<img src="Images/cost_plot_trained.svg" width="500">

### GUI

This application lets you/ the user draw on a canvas with its mouse and the trained network will guess the number.

On top you can train a new network yourself. Start by initializing the network with random values and then train it cycle by cycle. Whenever you want, you can check how the network performs on you drawn numbers. It's nice to see the networks' growth in confidence!

<img src="Images/gui_examples.png" width="700">

### Reference 3Blue1Brown

The mathematics and understanding behind the code and this network in general are based on the neural network series by 3Blue1Brown on YouTube.

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

### Mathematics & Theory

<img src="Images/network_image.png" width="900">

The MNIST dataset provides a 28 x 28 = 784 pixel grid which are used as input neurons. Each pixel/neuron represents how "painted" the pixel is from 0-1. The two hidden layer sizes $n_1 = n_2 = 16$ are arbitrary. The activation of the last layer $a^{(3)}$ represents how sure the network is about each number being the drawn one.

The activation $a^{(n)}$ of layer $n$ is calculated with the weight matrix $W^{(n)}$, its bias $b^{(n)}$ and the previous activation vector $a^{(n-1)}$ via the following equation.

$$a^{(n)} = \sigma \left( W^{(n)} \cdot a^{(n-1)} + b^{(n)} \right), \qquad n = 1,2,3$$

Our goal is to minimize the cost function $C$, which is a measure on how well the network performs. The smaller the better.

$$C = \sum_{k=1}^{n_3} \left(a_k^{(3)} - y_k \right)^2$$

$y$ is a vector that represents the actual drawn number. For example if the number is $3$, $y = (0,0,0,1,0,0,...)^{T} $.

To minimize $C$ we need formulas for the gradient $\nabla C$, which are are listed below. ($\sigma$ represents the sigmoid function.)

$$\frac{\partial C}{\partial w_{ij}^{(3)}} = 2 \left(a_i^{(3)} - y_i \right) \cdot \sigma^{\prime} \left(z_i^{(3)} \right) \cdot a_j^{(2)}$$

$$\frac{\partial C}{\partial b_{i}^{(3)}} = 2 \left(a_i^{(3)} - y_i \right) \cdot \sigma^{\prime} \left(z_i^{(3)} \right)$$

$$\frac{\partial C}{\partial w_{ij}^{(2)}} = \sigma^{\prime} \left(z_i^{(2)} \right) \cdot a_j^{(1)} \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot w_{ki}^{(3)}$$

$$\frac{\partial C}{\partial b_{i}^{(2)}} = \sigma^{\prime} \left(z_i^{(2)} \right) \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot w_{ki}^{(3)}$$

$$\frac{\partial C}{\partial w_{ij}^{(1)}} = \sigma^{\prime} \left(z_i^{(1)} \right) \cdot a_j^{(\text{in})} \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime} \left(z_l^{(2)} \right) \cdot w_{li}^{(2)}$$

$$\frac{\partial C}{\partial b_{i}^{(1)}} = \sigma^{\prime} \left(z_i^{(1)} \right) \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime} \left(z_l^{(2)} \right) \cdot w_{li}^{(2)}$$

$$\sigma(x) = \frac{1}{1+\text{e}^{-x}}$$
