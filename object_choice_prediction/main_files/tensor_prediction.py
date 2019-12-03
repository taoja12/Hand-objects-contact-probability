import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf

np.set_printoptions(threshold=np.inf)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

train_path = pd.read_csv("/home/tao/projects/cnns_data/tao_cnn_train.csv", sep=',')
train_data = train_path.iloc[:, 2:12].values
y = pd.get_dummies(train_path.label)  # eraser[10000],lotion[01000],square[00100],on_the_way[00010],tennis_ball[00001]
print(y)
test_path = pd.read_csv("/home/tao/projects/cnns_data/tao_cnn_test.csv", sep=',')
test_data = test_path.iloc[:, 2:12].values
test_label = pd.get_dummies(test_path.label)
print(test_label)
batch_size = 16
n_batch = len(train_data) // batch_size
# print(n_batch)  83

epsilon = 0.001


def batch_normal_layers(inputs, is_training, decay=0.999):
    gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_avg = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        avg, var = tf.nn.moments(inputs, [0])
        train_avg = tf.assign(pop_avg, pop_avg * decay + avg * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + var * (1 - decay))
        with tf.control_dependencies([train_avg, train_var]):
            return tf.nn.batch_normalization(inputs, avg, var, beta, gamma, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_avg, pop_var, beta, gamma, epsilon)


x = tf.placeholder(tf.float32, [None, 10])
y_ = tf.placeholder(tf.float32, [None, 5])

# layer1
w1 = tf.Variable(tf.zeros([10, 10]))
b1 = tf.Variable(tf.zeros([10]))
# z1 = tf.matmul(x, w1) + b1
z1 = batch_normal_layers((tf.matmul(x, w1) + b1), is_training=True)
l1 = tf.nn.sigmoid(z1)

# # layer2
# w2 = tf.Variable(tf.zeros([10, 10]))
# b2 = tf.Variable(tf.zeros([10]))
# z2 = tf.matmul(l1, w2) + b2
# # z2 = batch_normal_layers((tf.matmul(l1, w2) + b2), is_training=True)
# l2 = tf.nn.sigmoid(z2)

# softmax
w3 = tf.Variable(tf.zeros([10, 5]))
b3 = tf.Variable(tf.zeros([5]))
z = batch_normal_layers((tf.matmul(l1, w3) + b3), is_training=True)
# z = tf.matmul(l2, w3) + b3
prediction = tf.nn.softmax(z)

# loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(prediction), reduction_indices=[1]))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
loss = tf.reduce_mean(tf.square(y_ - prediction))
# loss = -tf.reduce_sum(y_ * tf.log(prediction))

# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# loss_file = open("1layer_bn_loss3_lr0.5.txt", "w")

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    for epoch in range(50000):
        sess.run(train_step, feed_dict={x: train_data, y_: y})
        # print(len(train_data))
        if epoch % 500 == 0:
            total_loss, acc = sess.run([loss, accuracy], feed_dict={x: train_data, y_: y})
            print("Iter %d, loss %g, train_accuracy %g" % (epoch, total_loss, acc))
            # loss_file.write("Iter %d, loss %g, train_accuracy %g" % (epoch, total_loss, acc) + '\n')

    test_acc, probs = sess.run([accuracy, prediction], feed_dict={x: test_data, y_: test_label})
    print("test accuracy %g" % test_acc)

    with open("tao_probs.csv", "w") as probsfile:
        writer = csv.writer(probsfile)
        for row in probs:
            writer.writerow(row)
