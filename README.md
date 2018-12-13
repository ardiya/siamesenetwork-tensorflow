# Siamese Network Tensorflow

Siamese network is a neural network that contain two or more identical subnetwork. The objective of this network is to find the similarity or comparing the relationship between two comparable things. Unlike classification task that uses cross entropy as the loss function, siamese network usually uses contrastive loss or triplet loss.

Siamese network has a lot of function, this repository is trying to use Siamese network to do a dimensionality reduction and image retrieval.

This project follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the output of the shared network and by optimizing the contrastive loss (see paper for more details). The contastive loss is defined as follows

<img src="https://latex.codecogs.com/gif.latex?\begin{align}&space;L_{contrastive}&space;&=&space;L_{similarity}&plus;L_{dissimilarity}&space;\notag&space;\\&space;&=&space;\frac{1}{2}(Y)(D)^2&plus;\frac{1}{2}(1-Y)(max(0,m-D))^2&space;\notag&space;\end{align}" title="\begin{align} L_{contrastive} &= L_{similarity}+L_{dissimilarity} \notag \\ &= \frac{1}{2}(Y)(D)^2+\frac{1}{2}(1-Y)(max(0,m-D))^2 \notag \end{align}" alt="-contrastive loss function-"/>

The <img src="https://latex.codecogs.com/gif.latex?D=\sqrt{(N(x_{left}-x_{right}))2}"  alt="-D formula-" /> is the distance of between the output of the network <img src="https://latex.codecogs.com/gif.latex?N"  alt="N" /> with the input <img src="https://latex.codecogs.com/gif.latex?x_{left}"  alt="Xleft" /> and the input <img src="https://latex.codecogs.com/gif.latex?x_{right}"  alt="Xright" />. 

The similarity function is defined as <img src="https://latex.codecogs.com/gif.latex?L_{similarity}=\frac{1}{2}(Y)(D)^2" alt="-sim function-" />. This function will be activated when the Label <img src="https://latex.codecogs.com/gif.latex?Y"  alt="Y" /> equal to 1 and deactivated when <img src="https://latex.codecogs.com/gif.latex?Y"  alt="Y" /> is equal to 0. The goal of this function is to minimize the distance of the pairs.

The dissimilarity function is defined as <img src="https://latex.codecogs.com/gif.latex?L_{similarity}=\frac{1}{2}(1-Y)(max(0,m-D))^2" alt="-dissim function-" />. This function will be activated when the Label <img src="https://latex.codecogs.com/gif.latex?Y"  alt="Y" /> is equal to 0 and deactivated when <img src="https://latex.codecogs.com/gif.latex?Y"  alt="Y" /> is equal to 1. The goal of this function is to give a penalty of the pairs when the distance is lower than margin <img src="https://latex.codecogs.com/gif.latex?m"  alt="m" />.

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

## Model
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/tensorboard-graph.png)
The input of these will be image_left, image_right and .
Our model uses 5 layer of convolutional layer and pooling followed. We do not use fully convolutonal net because convolution operation is faster on GPU(especially using CUDNN). See http://cs231n.github.io/convolutional-networks/#convert for more information on converting FC layer to Conv layer.

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

## Updates
- Update the API to 1.0
- Cleanup the old code

## Dimensionality reduction
The images below shows the final Result on MNIST test dataset. By only using 2 features, we can easily separate the input images.
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/result.jpg)

The gif below shows some animation until it somehow converges.
![](https://raw.githubusercontent.com/ardiya/siamesenetwork-tensorflow/master/figure/myfig.gif)

## Image retrieval
Image retrieval uses the trained model to extract the features and get the most similar image using cosine similarity.
[See here](https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/Similar%20image%20retrieval.ipynb "See the code here")

#### Retrieving similar test image from trainset
- Select id 865 in test image
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/random-test.png)

- Retrieved top n similar image from train data
with ids of [53144 47864 11074 51561 41350 34215 48182] from train data
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/retrieve-from-train.png)