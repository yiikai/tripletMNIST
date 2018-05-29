import tensorflow as tf
import numpy as np
from triplet_loss import *

import urllib
import urllib.request
import shutil
import gzip
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

tf.logging.set_verbosity(tf.logging.INFO)


def get_train_set():
    return (mnist.train.images, mnist.train.labels)


def get_test_set():
    return (mnist.test.images, mnist.test.labels)


x_train, y_train = get_train_set()

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int32)

input_x = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('Layer1'):
    out = tf.layers.conv2d(inputs=input_x, filters=32, kernel_size=3, strides=2, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                           bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    out = tf.nn.relu(out)
    out = tf.layers.max_pooling2d(out, 2, 2)

    out = tf.layers.conv2d(inputs=input_x, filters=64, kernel_size=3, strides=2, padding='same',
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                           bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    out = tf.nn.relu(out)
    out = tf.layers.max_pooling2d(out, 2, 2)
    print(out.shape)
    out = tf.reshape(out, [-1, 7 * 7 * 64])

with tf.name_scope('Layer2'):
    logits = tf.layers.dense(inputs=out, units=10, activation=None, use_bias=True,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

print('logits is ', logits.shape)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(loss)
loss_triplet = batch_hard_triplet_loss(y, logits, margin=1, squared=False)
logits = tf.nn.softmax(logits)
print("softmax logits:", logits)
predict = tf.argmax(logits, 1)
predict = tf.cast(predict, tf.int32)
print("predict value:", predict.shape)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))

optimizer = tf.train.AdamOptimizer(1e-3)
global_step = tf.train.get_global_step()
train_op = optimizer.minimize(loss, global_step=global_step)
train_op2 = optimizer.minimize(loss_triplet, global_step=global_step)
tf.summary.scalar('LOSS', loss)
tf.summary.tensor_summary('LOSS', loss)
merged = tf.summary.merge_all()

from sklearn import cross_validation

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.2, random_state=0)

trainsize = len(x_train)
print(trainsize)
batch_size = 440
epochs = 100

train_data = (x_train, y_train)
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import pylab
# one_pic_arr = np.reshape(x_train[1],(28,28))
# print(y_train[1])
# plt.imshow(one_pic_arr)
# pylab.show()


sess = tf.Session()
train_writer = tf.summary.FileWriter('./model/train', sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(epochs):
    # print("epoch:{}".format(i))
    t_x = x_train[i * batch_size:i * batch_size + batch_size]
    t_y = y_train[i * batch_size:i * batch_size + batch_size]

    _, loss_cost = sess.run([train_op, loss], feed_dict={x: t_x, y: t_y})
    #train_writer.add_summary(SM, i)
    if i % 10 == 0:
        print("loss_cost is:", loss_cost)


for i in range(epochs):
    t_x = x_train[i * batch_size:i * batch_size + batch_size]
    t_y = y_train[i * batch_size:i * batch_size + batch_size]
    _ ,loss_tip = sess.run([train_op2,loss_triplet], feed_dict={x: t_x, y: t_y})
    if i % 10 == 0:
        print(":loss_trip is:",loss_tip)

print("Train OVER!!!!!!!!!!")

print("Start Eval!!!!!!!!!")
acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
print("acc is:", acc)
