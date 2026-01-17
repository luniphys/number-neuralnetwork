# Neural network from scratch

This network will detect drawn numbers based on the MNIST dataset. In this code I will only implement basic Python functions and math (no nerual network/AI packages).

The mathematics and understanding behind the code and this network in general are based on the neural network series by 3Blue1Brown on YouTube.

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

<br>
<br>

The formulas for the gradient $\nabla C$ are listed below.

$$\frac{\partial C}{\partial w_{ij}^{(3)}} = 2 (a_i^{(3)} - y_i ) \cdot \sigma^{\prime}(z_i^{(3)}) \cdot a_j^{(2)}$$

$$\frac{\partial C}{\partial b_{i}^{(3)}} = 2 (a_i^{(3)} - y_i ) \cdot \sigma^{\prime}(z_i^{(3)})$$


$$\frac{\partial C}{\partial w_{ij}^{(2)}} = \sigma^{\prime}(z_i^{(2)}) \cdot a_j^{(1)} \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot w_{ki}^{(3)} $$

$$\frac{\partial C}{\partial b_{i}^{(2)}} = \sigma^{\prime}(z_i^{(2)}) \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot w_{ki}^{(3)} $$


$$\frac{\partial C}{\partial w_{ij}^{(1)}} = \sigma^{\prime}(z_i^{(1)}) \cdot a_j^{(\text{in})} \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime}(z_l^{(2)}) \cdot w_{li}^{(2)} $$

$$\frac{\partial C}{\partial b_{i}^{(1)}} = \sigma^{\prime}(z_i^{(1)}) \cdot \sum_{k=1}^{n_3} 2 (a_k^{(3)} - y_k ) \cdot \sigma^{\prime}(z_k^{(3)}) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime}(z_l^{(2)}) \cdot w_{li}^{(2)} $$



<br>
<br>
<br>

(Since training the weights and biases takes a couple of hours, an already trained set is provided.) 
