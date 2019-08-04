## coding: UTF-8
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import os
import pandas as pd

NUM_CLASSES = 2 # the number of classification
IMG_H_SIZE = 40 # height of image
IMG_W_SIZE = 40 # width of image

def get_MNIST_data():

    # directory of image
    img_dirs = ['iekei', 'jiro']

    # image
    image = []
    # label
    label = []

    img2 = np.zeros((3*IMG_H_SIZE, IMG_W_SIZE))

    for i, d in enumerate(img_dirs):
        # get file in ./data/…
        files = os.listdir('./Ramen_Data/' + d)
        for f in files:
            # 画像読み込み
            img = cv2.imread('./Ramen_Data/' + d + '/' + f)
            # resize to IMG_H_SIZE * IMG_W_SIZE and convert RGB data into 2D data
            img = cv2.resize(img, dsize=(IMG_W_SIZE, IMG_H_SIZE))
            for ii in range(IMG_H_SIZE):
                for jj in range(IMG_W_SIZE):
                    for kk in range(3):
                        img2[ii+IMG_H_SIZE*kk][jj] = img[ii][jj][kk]

            img2 = img2.astype(np.float32) / 255.0
            image.append(img2)
            # one_hot_vector(label)
            tmp = np.zeros(NUM_CLASSES)
            tmp[i] = 1
            label.append(tmp)

    # to numpy array
    image = np.asarray(image)
    label = np.asarray(label)

    X = image
    y = label

    # sort out train_data from test_data
    (train_x, test_x , one_hots_train, one_hots_test) \
        = train_test_split(
        X, y, test_size=0.1, random_state=5
    )

    train_x = train_x[:, :, :, np.newaxis]
    test_x = test_x[:, :, :,  np.newaxis]

    return train_x, one_hots_train, test_x, one_hots_test

def plot_MNIST(x, one_hot):

    row = 4
    column = 4
    p = random.sample(range(1, 80), row * column)

    plt.figure()

    for i in range(row * column):

        image = x[p[i]].reshape(IMG_H_SIZE*3,IMG_W_SIZE,1)
        image = image[:,:,0]
        plt.subplot(row, column, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title('label = {}'.format(np.argmax(one_hot[p[i]]).astype(int)))
        plt.axis('off')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                    wspace=0.05, hspace=0.3)
    plt.show()

def dense(input, name, in_size, out_size, activation="relu"):

    with tf.variable_scope(name, reuse=False):
        w = tf.get_variable("w", shape=[in_size, out_size],
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable("b", shape=[out_size], initializer=tf.constant_initializer(0.0))

        l = tf.add(tf.matmul(input, w), b)

        if activation == "relu":
            l = tf.nn.relu(l)
        elif activation == "sigmoid":
            l = tf.nn.sigmoid(l)
        elif activation == "tanh":
            l = tf.nn.tanh(l)
        else:
            l = l
        print(l)
    return l

def scope(y, y_, learning_rate=0.1):

    #Learning rate
    learning_rate = tf.Variable(learning_rate,  trainable=False)

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y, logits=y_), name="loss")

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       name="optimizer").minimize(loss)

    # Evaluate the model
    correct = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int32),
                       tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    #  Tensorboard
    writer = tf.summary.FileWriter('./Tensorboard/')
    # run this command in the terminal to launch tensorboard:
    # tensorboard --logdir=./Tensorboard/
    writer.add_graph(graph=sess.graph)

    return loss, accuracy, optimizer, writer

def confusion_matrix(cm, accuracy):
    plt.figure(figsize=(NUM_CLASSES, NUM_CLASSES))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=10)

train_x, one_hots_train, test_x, one_hots_test = get_MNIST_data()
number_test = [one_hots_test[i, :].argmax() for i in range(0, one_hots_test.shape[0])]

plot_MNIST(x=train_x, one_hot=one_hots_train)

n_label = len(np.unique(number_test))   # Number of class
height = train_x.shape[1]
width = train_x.shape[2]

# Session and context manager
tf.reset_default_graph()
sess = tf.Session()

with tf.variable_scope(tf.get_variable_scope()):

    # Placeholders
    x = tf.placeholder(tf.float32, [None, height, width, 1], name='X')
    y = tf.placeholder(tf.float32, [None, n_label], name='Y')


    # Convolutional Neural network
    c1 = tf.layers.conv2d(inputs=x, kernel_size=[5, 5], strides=[1, 1],
                          filters=16, padding='SAME', activation=tf.nn.relu,
                          name='Conv_1')
    print(c1)
    c1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2],
                                 strides=[2, 2], padding='SAME')
    print(c1)

    c2 = tf.layers.conv2d(inputs=c1, kernel_size=[3, 3], strides=[1, 1],
                          filters=32, padding='SAME', activation=tf.nn.relu,
                          name='Conv_2')
    print(c2)
    c2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2],
                                 strides=[2, 2], padding='SAME')
    print(c2)

    c3 = tf.layers.conv2d(inputs=c2, kernel_size=[3, 3], strides=[1, 1],
                          filters=64, padding='SAME', activation=tf.nn.relu,
                          name='Conv_3')
    print(c3)
    c3 = tf.layers.max_pooling2d(inputs=c3, pool_size=[2, 2],
                                 strides=[2, 2], padding='SAME')
    print(c3)

    c4 = tf.layers.conv2d(inputs=c3, kernel_size=[3, 3], strides=[1, 1],
                          filters=32, padding='SAME', activation=tf.nn.relu,
                          name='Conv_4')
    print(c4)
    c4 = tf.layers.max_pooling2d(inputs=c4, pool_size=[2, 2],
                                 strides=[2, 2], padding='SAME')
    print(c4)

    c5 = tf.layers.conv2d(inputs=c4, kernel_size=[3, 3], strides=[1, 1],
                          filters=32, padding='SAME', activation=tf.nn.relu,
                          name='Conv_5')
    print(c5)
    c5 = tf.layers.max_pooling2d(inputs=c5, pool_size=[2, 2],
                                 strides=[2, 2], padding='SAME')
    print(c5)

    # Reshape to a fully connected layers
    size = c5.get_shape().as_list()

    l1 = tf.reshape(c5, [-1, size[1] * size[2] * size[3]],
                    name='reshape_to_fully')
    print(l1)

    l2 = dense(input=l1, name="output_layers",
               in_size=l1.get_shape().as_list()[1], out_size=n_label,
               activation='None')

    # Softmax layer
    y_ = tf.nn.softmax(l2, name='softmax')
    print(y_)

    # Scope
    loss, accuracy, optimizer, writer = scope(y, y_, learning_rate=0.0001)

    # Initialize the Neural Network
    sess.run(tf.global_variables_initializer())

    # Train the Neural Network
    loss_history = []
    acc_history = []
    epoch = 100
    train_data = {x: train_x, y: one_hots_train}

    for e in range(epoch):

        _, l, acc = sess.run([optimizer, loss, accuracy], feed_dict=train_data)

        loss_history.append(l)
        acc_history.append(acc)

        print("Epoch " + str(e) + " - Loss: " + str(l) + " - " + str(acc))

plt.figure()
plt.plot(acc_history)
plt.xlabel('Epoch', fontsize = 14)
plt.ylabel('Acuuracy', fontsize=14)

# Test the trained Neural Network
test_data = {x: test_x, y: one_hots_test}
l, acc = sess.run([loss, accuracy], feed_dict=test_data)
print("Test - Loss: " + str(l) + " - " + str(acc))
predictions = y_.eval(feed_dict=test_data, session=sess)
predictions_int = (predictions == predictions.max(axis=1, keepdims=True)).astype(int)
predictions_numbers = [predictions_int[i, :].argmax() for i in range(0, predictions_int.shape[0])]

# Confusion matrix
cmm = metrics.confusion_matrix(number_test, predictions_numbers)
cm =pd.DataFrame(data=cmm, index=['Iekei','Jiro'], columns=['Iekei', 'Jiro'])
print(cm)
confusion_matrix(cm=cm, accuracy=acc)
cmN = cm / cm.sum(axis=0)
confusion_matrix(cm=cmN, accuracy=acc)
