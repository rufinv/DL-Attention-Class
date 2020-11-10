# DL-Attention-Class
## **Simple implementation of a ConvNet with attention**
* Using Keras, we will design a simple ConvNet (3 layers) and train it for classification on  the Cifar10 natural images dataset. 

* Without any attention, the model has about 2.5M parameters, and can reach 83% accuracy (if we use data augmentation during training). This is not state-of-the-art, but good enough for such a simple model.

* We will implement a **single-head attention layer** acting on the last convolutional layer. This will allow the network to match or compare features across spatial locations, and will bring down the number of output channels and consequently, the total number of parameters: 0.5M parameters. We will verify that the network can achieve about the same or better level of accuracy (83%)with 5x fewer parameters.

* We will visualize **attention maps** for a given location in an image, to see what attention features have been learned.

### Optional follow-up work
* We can verify that without attention, a ConvNet with 0.5M parameters would perform significantly worse (around 80%).

* We can implement **multi-head attention** to further improve performance (about 83.5%) with the same number of parameters (<0.6M parameters).

* We can run the models on Cifar100, for a more challenging task
