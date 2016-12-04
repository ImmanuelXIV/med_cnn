# Main draft to run the CNN on the medical data set
from sklearn.cross_validation import KFold
import utils
import tensorflow as tf
import random
import pickle

# ToDo:
# implement more text pre-processing (see utils.py -> check which words are not recognized by word2vec model)
# experiment with network architecture (e.g. number and size of layers, to get feeling what works)
# implement final grid-search with diff. hyper parameters
# report best setup for data and store results


# Set parameters
num_iter = 2000         # number of iterations per k-fold cross validation
n_gram = 3              # try different n-grams: 1, 2, 3, 4, 5
batch_size = 2          # how many sentences to feed the network at once
num_folds = 2           # how many folds in k-fold-cross-validation
data_subset = 8         # create a subset of size data_subset (get results faster)
eval_acc_every = 10     # in the training: evaluate the test accuracy ever X steps
num_classes = 5         # how many different classes exist?
keep_probability = 0.5  # this is the dropout (keep) value for the training of the CNN

# ToDo: Download the German word2vec model (700MB, link) and set the path accordingly
# https://tubcloud.tu-berlin.de/public.php?service=files&t=dc4f9d207bcaf4d4fae99ab3fbb1af16
model = "/home/immanuel/ETH/data/german.model"
diagnoses = "/home/immanuel/Desktop/sample10.txt"           # this is a txt file, where each line is a diagnosis
labels = "/home/immanuel/Desktop/sample10_lables.txt"       # this is the corresponding txt file, 1 class in each line

# Since padding='VALID' for the filter, dimensionality has to be reduced
reduce_dim = n_gram - 1

# Pre-process (clean, shuffle etc.) and load the medical data (respectively a subset of size data_subset)
data, labels, sequence_length = utils.load_data(model, diagnoses, labels, data_subset, num_classes)

# Initiate Tensorflow session
sess = tf.InteractiveSession()

# Create placeholders
# sequence_length = max word length of diagnoses (if diagnose is shorter -> augmented to max size with "PAD" word)
# 300 is dimensionality of each word embedding by pre-trained word2vec model on German Wikipedia data
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
W_conv1 = weight_variable([n_gram, 300, 1, 32])
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
print("Data Size: %d" % data_size)

kf = KFold(data_size, n_folds=num_folds, shuffle=False)

# Store final accuracy of different folds
count = 0
train_accuracy_fold = []
train_accuracy_total = []
test_accuracy_final = []

for train_index, test_index in kf:
    # Initialize variables
    sess.run(tf.initialize_all_variables())

    train_accuracy_fold = []
    # Split in train and test set
    count += 1
    print("TRAIN and TEST in Fold: %d" % count)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Check shape of arrays
    # print("Shape of arrays: ")
    # print(x_train.shape)
    # print(y_train.shape)

    # Start training loop
    for i in range(num_iter):
        # Set batch size (how many diff. diagnoses presented at once)
        indices = random.sample(xrange(len(x_train)), batch_size)
        batch = x_train[indices]
        ybatch = y_train[indices]

        if i % eval_acc_every == 0:                         # evaluate every X-th training step to monitor overfitting
            # run the evaluation step
            train_accuracy = accuracy.eval(feed_dict={
                x: batch, y_: ybatch, keep_prob: 1.0})      # in evaluation dropout has to be 1.0

            # save value
            train_accuracy_fold.append(train_accuracy)
            print("step %d, training accuracy %g" % (i, train_accuracy))

        # run the training step
        train_step.run(feed_dict={x: batch, y_: ybatch, keep_prob: keep_probability}) # in training dropout can be set

    test_acc = accuracy.eval(feed_dict={
        x: x_test, y_: y_test, keep_prob: 1.0})

    # Store
    train_accuracy_total.append(train_accuracy_fold)
    test_accuracy_final.append(test_acc)
    print("TEST ACCURACY %g" % test_acc)

print("\nFINAL TEST ACCURACY per k-fold: ")
print(test_accuracy_final)

print("\nTRAIN ACCURACIES all: ")
print(train_accuracy_total)

# Store TRAIN results in pickle files
"""
f = open('./Results/' + str(data_subset) + '_' + str(NIter) + '_' + str(N_gram) + '_' + str(NFolds) + '_results_Train.pckl', 'w')
pickle.dump(train_accuracy_total, f)
f.close()

# Store TEST results in pickle files
f = open('./Results/' + str(data_subset) + '_' + str(NIter) + '_' + str(N_gram) + '_' + str(NFolds) + '_results_Test.pckl', 'w')
pickle.dump(test_accuracy_final, f)
f.close()
"""
