#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn import model_selection
import time as t

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# scale data
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


NUM_FEATURES = 21
NUM_CLASSES = 3
NUM_HIDDEN = 10

learning_rate = 0.01
weight_decay_param = 10 ^ (-6)
epochs = 1000
batch_size = 32

seed = 10
np.random.seed(seed)

# read train data

train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter=',')


def soft_max_classification(input_data):
    hot_mat = np.zeros((input_data.shape[0], NUM_CLASSES))
    hot_mat[np.arange(input_data.shape[0]), input_data - 1] = 1  # one hot matrix
    return hot_mat


def init_bias(n=1):
    return tf.Variable(np.zeros([n]), dtype=tf.float32, name='biases')


def init_weights(n_in=1, n_out=1):
    return tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0 / math.sqrt(float(n_in))), dtype=tf.float32,
                       name='weights')


# Split dataset into train / test
trainX, testX, trainY, testY = model_selection.train_test_split(
    train_input[1:, :21], train_input[1:, -1].astype(int), test_size=0.3, shuffle=True)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
testX = scale(testX, np.min(testX, axis=0), np.max(testX, axis=0))
print(trainX)

totalCount = len(trainX)

trainY = soft_max_classification(trainY)
testY = soft_max_classification(testY)

n = trainX.shape[0]

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
d = tf.placeholder(tf.float32, [None, NUM_CLASSES])

print(x.shape)
print(d.shape)

# Build the graph for the deep net

# Define variables:
V = init_weights(NUM_HIDDEN, NUM_CLASSES)
c = init_bias(NUM_CLASSES)
W = init_weights(NUM_FEATURES, NUM_HIDDEN)
b = init_bias(NUM_HIDDEN)
#
z = tf.matmul(x, W) + b
h = tf.nn.relu(z)
u = tf.matmul(h, V) + c
p = tf.exp(u) / tf.reduce_sum(tf.exp(u), axis=1, keepdims=True)
y = tf.argmax(p, axis=1)

grad_u = -(d - p)
grad_V = tf.matmul(tf.transpose(h), grad_u)
grad_c = tf.reduce_sum(grad_u, axis=0)

dh = h * (1 - h)
grad_z = tf.matmul(grad_u, tf.transpose(V)) * dh

grad_W = tf.matmul(tf.transpose(x), grad_z)
grad_b = tf.reduce_sum(grad_z, axis=0)

W_new = W.assign(W - learning_rate * (grad_W + weight_decay_param * W))
b_new = b.assign(b - learning_rate * grad_b)
V_new = V.assign(V - learning_rate * (grad_V + weight_decay_param * V))
c_new = c.assign(c - learning_rate * grad_c)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=d, logits=u)
loss = tf.reduce_mean(cross_entropy)

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

correct_prediction = tf.cast(tf.equal(tf.argmax(u, 1), tf.argmax(d, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    start_time = t.time()
    for i in range(epochs):
        batch_acc = []
        for j in range(batch_size, totalCount, batch_size):
            batch_x = trainX[j - batch_size:j]
            batch_y = trainY[j - batch_size:j]
            fold_acc = [];
            for k in range(5):
                start_index = k * (batch_size // 5)
                stop_index = (k + 1) * (batch_size // 5)
                train_x = np.concatenate((batch_x[0:start_index],batch_x[stop_index:batch_size]))
                train_y = np.concatenate((batch_y[0:start_index],batch_y[stop_index:batch_size]))
                train_op.run(feed_dict={x: train_x, d: train_y})
                fold_acc.append(
                    accuracy.eval(feed_dict={x: batch_x[start_index:stop_index], d: batch_y[start_index:stop_index]}))
            batch_acc.append(sum(fold_acc)/len(fold_acc))
        # train_op.run(feed_dict={x: trainX, d: trainY})
        # train_acc.append(accuracy.eval(feed_dict={x: trainX, d: trainY}))
        train_acc.append(sum(batch_acc) / len(batch_acc))
        if i % 100 == 0:
            print('iter %d: accuracy %g' % (i, train_acc[i]))
    stop_time = t.time()
    print(-start_time + stop_time)
    print(accuracy.eval(feed_dict={x: testX, d: testY}))

# plot learning curves

plt.figure(1)
plt.plot(range(epochs), train_acc)
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Train accuracy')
plt.show()
