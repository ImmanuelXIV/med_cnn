# Main file to run the CNN on the Twitter data set
# to perform the sentiment analysis

from sklearn.cross_validation import KFold
from numpy import genfromtxt
import utils
import tensorflow as tf
import numpy as np
import random
import pickle
#import matplotlib.pyplot as plt


# Set parameters
NIter = 2000
N_gram = 3
BatchSize = 1
NFolds = 2
# Path = './GoogleNews-vectors-negative300.bin'     # DOWNLOAD THIS FILE (1.8GB): https://github.com/3Top/word2vec-api
model = "/home/immanuel/ETH/data/german.model"
# PosTweets = "./data/50Kpositive.txt"              # 50.000 positive Tweets
# NegTweets = "./data/50Knegative.txt"              # 50.000 negative Tweets
data_subset = 8     # create data subset


num_classes = 5
diagnoses = "/home/immanuel/Desktop/sample10.txt"
labels = "/home/immanuel/Desktop/sample10_lables.txt"

# Since padding='VALID' for the filter, dimensionality has to be reduced
reduce_dim = N_gram-1

# Pre-process (shuffle, clean etc.) and load the twitter data (or a subset)
data, labels, sequence_length = utils.load_data(model, diagnoses, labels, data_subset)

# Initiate Tensorflow session
sess = tf.InteractiveSession()

# Create placeholders
# sequence_length = max tweet word length (if tweet is shorter -> augmented to max size)
# 300 is dimensionality of each word embedding by pre-trained word2vec model on German Wikipedia data
# Link to download: https://tubcloud.tu-berlin.de/public.php?service=files&t=dc4f9d207bcaf4d4fae99ab3fbb1af16
# on http://devmount.github.io/GermanWordEmbeddings/,
# 5 output neurons for 0-4 cancer stadiums
x = tf.placeholder("float", shape=[None, sequence_length, 300, 1])
y_ = tf.placeholder("float", shape=[None, num_classes])

# Initialize variables
sess.run(tf.initialize_all_variables())


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')    # do not change VALID

# First Convolutional Layer
W_conv1 = weight_variable([N_gram, 300, 1, 32])
b_conv1 = bias_variable([32])

# Convolve with the weight tensor, add the bias, apply the ReLU function
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

# Densely Connected Layer
# reduce_dim = 3-gram -1 = 2
W_fc1 = weight_variable([(sequence_length-reduce_dim)*32, 1024])
b_fc1 = bias_variable([1024])

# Reshape input for the last layer
h_pool2_flat = tf.reshape(h_conv1, [-1, (sequence_length-reduce_dim)*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer (FC -> 5 neurons)
W_fc2 = weight_variable([1024, num_classes])
b_fc2 = bias_variable([num_classes])

# Apply softmax
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

print('Perform cross-validation...')
data_size = len(labels)
kf = KFold(data_size, n_folds=NFolds, shuffle=False)

# Store final accuracy of folds
count = 0
train_accuracy_fold = []
train_accuracy_total = []
test_accuracy_final = []

for train_index, test_index in kf:
    # Initialize variables again
    sess.run(tf.initialize_all_variables())

    train_accuracy_fold = []
    # Split in train and test set
    count += 1
    print("TRAIN and TEST in Fold: %d" % count)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Loop over number iterations
    for i in range(NIter):
        # Set batch size
        indices = random.sample(xrange(len(x_train)), BatchSize)
        batch = x_train[indices]
        ybatch = y_train[indices]

        """
        if i % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch, y_: ybatch, keep_prob: 1.0})

            # save value
            train_accuracy_fold.append(train_accuracy)
            print("step %d, training accuracy %g" % (i, train_accuracy))
        """
        train_step.run(feed_dict={x: batch, y_: ybatch, keep_prob: 0.5})

    test_acc = accuracy.eval(feed_dict={
        x: x_test, y_: y_test, keep_prob: 1.0})

    # Store
    train_accuracy_total.append(train_accuracy_fold)
    test_accuracy_final.append(test_acc)
    print("TEST ACCURACY %g" % test_acc)

print("\nFINAL TEST ACCURACY: ")
print(test_accuracy_final)

print("\nTRAIN ACCURACIES TOTAL: ")
print(train_accuracy_total)

# Store TRAIN results in pickle files
f = open('./Results/' + str(data_subset) + '_' + str(NIter) + '_' + str(N_gram) + '_' + str(NFolds) + '_results_Train.pckl', 'w')
pickle.dump(train_accuracy_total, f)
f.close()

# Store TEST results in pickle files
f = open('./Results/' + str(data_subset) + '_' + str(NIter) + '_' + str(N_gram) + '_' + str(NFolds) + '_results_Test.pckl', 'w')
pickle.dump(test_accuracy_final, f)
f.close()

# Open results in pickle files
#f = open('./Results/' + str(DataSubSet) + '_' + str(NIter) + '_' + str(N_gram) + '_' + str(NFolds) + '_results_Train.pckl')
#obj_train = pickle.load(f)
#f.close()

#f = open('./Results/' + str(DataSubSet) + '_' + str(NIter) + '_' + str(N_gram) + '_' + str(NFolds) + '_results_Test.pckl')
#obj_test = pickle.load(f)
#f.close()

# Plot curves etc.