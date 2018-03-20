#programming model, a grah with nodes and tensor which are operations
#Variables are blue in the graph
#Placeholder are green. This is data or bathces of data

#Once we define an optimizer and a loss function, all nodes will know how to perform ackprop

#sess.run(tf.initialize_all_variable)  initialize the variable just declarated before
#see it is deprecated, look for the new initilaizer

#x= placeholder...... name='x') the name is the one which will be displayed in the tensorboard

import tensorflow as tf
import numpy as np

# MatMul multiply two matrices
# Add add elementwise (with broadcasting)
# ReLU activate with elementwise rectified linear function

# Variables
b= tf.Variable(tf.zeros((100,)))
W= tf.Variable(tf.random_uniform((784, 100), -1, 1)) #Weights initalization uniform
# Input
x= tf.placeholder(tf.float32, (100,784))
# Activation function
h= tf.nn.relu(tf.matmul(x,W)+b)

#tf.get_default_graph().get_operations()

# Create a session
sess = tf.Session()
#sess.run(fetches, feeds)
# Fetches is a list of graph nodes. Return the outputs of these nodes
# Feeds is a dictionary mapping from graph nodes to concrete values. Specifices the value of
# each graph node given in the dictionary
sess.run(tf.initialize_all_variables())
#give as input the input and the activation function
sess.run(h, {x:np.random.random(100,784)})


# We first built a graph using variables and placeholders
# We then deployed the graph onto a session, which is the execution environment
# Next comes hoe to train a model


prediction = tf.nn.softmax(...)
label = tf.placeholder(tf.float32, [100,10])
cross_entropy = -tf.reduce_mean(-tf.reduce_sum(label*tf.log(prediction)), axis=1)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
	batch_x, batch_label = data.next_batch()
	sess.run(train_step, feed_dict={x: batch_x, label: batch_label})

#Variable sharing

variables_dict = {
	"weights": tf.Variable(tf.random_normal([782,100]), name="weights")
	"biases": tf.Variable(tf.zeros([100]), name="biases")
} # ALl this is NOT GOOD for encapsulation !!!

with tf.variable_scope("foo"): #provides simple name-spacing to avoid clashes
     v=tf.get_variable("v", shape=[1]) #v.name == "foo/v:0"
with tf.get_variable_scope("foo", reuse=True): #Shared variable found
     v=tf.get_variable("v");
with tf.get_variable_scope("foo", reuse=False): #CRASH foo/v: already exists!
     v=tf.get_variable("v");


#To summarize
#1.Build a graph
#a. Feedforward / Prediction
#b. Optimization (gradients and train_step operation)
#2. Initialize a session
#3. Train with session.run(train_step, feed_dict)

