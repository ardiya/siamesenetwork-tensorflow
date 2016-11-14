#Siamese Network Tensorflow
Siamese network is a neural network that contain two or more identical subnetwork. The purpose of this network is to find the similarity or comparing the relationship between two comparable things. Unlike classification task that uses cross entropy as the loss function, siamese network usually uses contrastive loss or triplet loss.

This project follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the output of the shared network and by optimizing the contrastive loss (see paper for more details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
## Model
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/tensorboard-graph.png)
Our model uses 5 layer of convolutional layer and pooling followed. We do not use fully convolutonal net because convolution operation is faster on GPU(especially using CUDNN). See http://cs231n.github.io/convolutional-networks/#convert for more information on converting FC layer to Conv layer.

## Dimensionality reduction
Result on MNIST Dataset:
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/result.jpg)
See folder img to see the process until it converge, it is really fun to watch :)
## Image retrieval
TODO: create .ipynb file
## Run
Train the model
```bash
git clone https://github.com/ardiya/siamesenetwork-tensorflow
python train.py
```
Tensorboard Visualization(After training)
```bash
tensorboard --logdir=train.log
```