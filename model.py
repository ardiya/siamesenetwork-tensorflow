import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

flags = tf.app.flags
FLAGS = flags.FLAGS

def mynet(input, reuse=False):
	with tf.name_scope("model"):
		with tf.variable_scope("conv1") as scope:
			net = tf.contrib.layers.conv2d(input, 32, [7, 7], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv2") as scope:
			net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv3") as scope:
			net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv4") as scope:
			net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		with tf.variable_scope("conv5") as scope:
			net = tf.contrib.layers.conv2d(net, 2, [1, 1], activation_fn=None, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
			net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

		net = tf.contrib.layers.flatten(net)
	#	net = tf.contrib.layers.fully_connected(net, 2, activation_fn=None)

	return net

# def net(input_left, input_right):
# 	#Variables
# 	conv1_weights = tf.get_variable("conv1_weights", shape=[5, 5, 1, 20],
# 		  initializer=tf.contrib.layers.xavier_initializer())
# 	tf.histogram_summary("conv1/w", conv1_weights)
# 	conv1_biases = tf.Variable(tf.zeros([20]), name="conv1_biases")
# 	tf.histogram_summary("conv1/b", conv1_biases)

# 	conv2_weights = tf.get_variable("conv2_weights", shape=[5, 5, 20, 50],
# 		  initializer=tf.contrib.layers.xavier_initializer())
# 	tf.histogram_summary("conv2/w", conv2_weights)
# 	conv2_biases = tf.Variable(tf.zeros([50]), name="conv2_biases")
# 	tf.histogram_summary("conv2/b", conv2_biases)

# 	fc1_weights = tf.get_variable("fc1_weights", shape=[7*7*50, 1000],
# 		  initializer=tf.contrib.layers.xavier_initializer())
# 	tf.histogram_summary("fc1/w", fc1_weights)
# 	fc1_biases = tf.Variable(tf.zeros([1000]), name="fc1_biases")
# 	tf.histogram_summary("fc1/b", fc1_biases)

# 	fc2_weights = tf.get_variable("fc2_weights", shape=[1000, 2],
# 		  initializer=tf.contrib.layers.xavier_initializer())
# 	tf.histogram_summary("fc2/w", fc2_weights)
# 	fc2_biases = tf.Variable(tf.zeros([2]), name="fc2_biases")
# 	tf.histogram_summary("fc2/b", fc2_biases)

# 	#Left network
# 	with tf.name_scope("conv1"):
# 		conv1 = tf.nn.conv2d(input_left, conv1_weights,
# 			strides=[1, 1, 1, 1], padding='SAME', name="conv1")
# 		conv1 = tf.nn.relu(conv1 + conv1_biases, name="conv1_act")
# 		conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
# 						strides=[1, 2, 2, 1], padding='SAME', name="pool1")
# 	with tf.name_scope("conv2"):
# 		conv2 = tf.nn.conv2d(conv1, conv2_weights,
# 			strides=[1, 1, 1, 1], padding='SAME', name="conv2")
# 		conv2 = tf.nn.relu(conv2 + conv2_biases, name="conv2_act")
# 		conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
# 						strides=[1, 2, 2, 1], padding='SAME', name="pool2")
# 	flatten = tf.reshape(conv2, [-1, 7*7*50], name="flatten")
# 	with tf.name_scope("fc1"):
# 		fc1 = tf.matmul(flatten, fc1_weights) + fc1_biases
# 	with tf.name_scope("fc2"):
# 		fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases

# 	#Right net
# 	with tf.name_scope("conv1_p"):
# 		conv1_p = tf.nn.conv2d(input_right, conv1_weights,
# 			strides=[1, 1, 1, 1], padding='SAME', name="conv1_p")
# 		conv1_p = tf.nn.relu(conv1_p + conv1_biases, name="conv1_act_p")
# 		conv1_p = tf.nn.max_pool(conv1_p, ksize=[1, 2, 2, 1],
# 						strides=[1, 2, 2, 1], padding='SAME', name="pool1_p")
# 	with tf.name_scope("conv2_p"):
# 		conv2_p = tf.nn.conv2d(conv1_p, conv2_weights,
# 			strides=[1, 1, 1, 1], padding='SAME', name="conv2_p")
# 		conv2_p = tf.nn.relu(conv2_p + conv2_biases, name="conv2_act_p")
# 		conv2_p = tf.nn.max_pool(conv2_p, ksize=[1, 2, 2, 1],
# 						strides=[1, 2, 2, 1], padding='SAME', name="pool2_p")
# 	flatten_p = tf.reshape(conv2_p, [-1, 7*7*50], name="flatten_p")
# 	with tf.name_scope("fc1_p"):
# 		fc1_p = tf.matmul(flatten_p, fc1_weights) + fc1_biases
# 	with tf.name_scope("fc2_p"):
# 		fc2_p = tf.matmul(fc1_p, fc2_weights) + fc2_biases


# 	return fc2, fc2_p

def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		d = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1, model2), 2), 1, keep_dims=True))
		tmp= y * tf.square(d)    
		tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
		return tf.reduce_mean(tmp + tmp2) /2
