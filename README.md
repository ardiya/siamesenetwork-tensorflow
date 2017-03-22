#Siamese Network Tensorflow
Siamese network is a neural network that contain two or more identical subnetwork. The purpose of this network is to find the similarity or comparing the relationship between two comparable things. Unlike classification task that uses cross entropy as the loss function, siamese network usually uses contrastive loss or triplet loss.

This project follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the output of the shared network and by optimizing the contrastive loss (see paper for more details).

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
- Model provided is from iter-5000, the images below are trained until iter-50k 

## Dimensionality reduction
Result on MNIST Dataset:
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/result.jpg)
See folder [img](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/img "img") to see the process until it converge, it is really fun to watch :)

## Image retrieval
Image retrieval uses the trained model to extract the features and get the most similar image using cosine similarity.
[See here](https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/Similar%20image%20retrieval.ipynb "See the code here")

#### Retrieving similar test image from trainset
- Select id 865 in test image
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/random-test.png)

- Retrieved top n similar image from train data
with ids of [53144 47864 11074 51561 41350 34215 48182] from train data
![](https://github.com/ardiya/siamesenetwork-tensorflow/raw/master/figure/retrieve-from-train.png)