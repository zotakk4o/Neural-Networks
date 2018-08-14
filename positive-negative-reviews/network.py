import data_handlert
import os
import pickle
import tensorflow as tf


class Network():
    def __init__(self, hidden_layers, epochs, batch_size, pickle_name):
        self.train_x, self.train_y, self.test_x, self.test_y = self.read_pickled_data(pickle_name)
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = []

        sizes = [len(self.train_x[0])]
        sizes += hidden_layers
        sizes.append(2)

        self.sizes = sizes

        for i in range(1, len(sizes)):
            layer = {'weights': tf.Variable(tf.random_normal([sizes[i - 1], sizes[i]])),
                     'biases': tf.Variable(tf.random_normal([sizes[i]]))}
            self.layers.append(layer)

    def feed_forward(self, a):
        for layer in self.layers:
            if layer != self.layers[-1]:
                a = tf.nn.relu(tf.add(tf.matmul(a, layer['weights']), layer['biases']))
            else:
                a = tf.add(tf.matmul(a, layer['weights']), layer['biases'])

        return a

    def train(self):
        x = tf.placeholder(float, [None, self.sizes[0]])
        y = tf.placeholder(float)

        prediction = self.feed_forward(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                for i in range(0, len(self.train_x), self.batch_size):
                    batch_x = self.train_x[i:i + self.batch_size]
                    batch_y = self.train_y[i:i + self.batch_size]
                    sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, float))
                print(
                    f'Accuracy: {int(accuracy.eval({x: self.test_x, y: self.test_y})* 100)}%, Epoch: {epoch + 1}')

    def read_pickled_data(self, pickle_file):
        file = open(pickle_file, 'rb')
        train_x, train_y, test_x, test_y = pickle.load(file, encoding = 'latin1')
        file.close()
        return train_x, train_y, test_x, test_y


pickle_name = 'classified_data.pickle'

if not os.path.isfile(pickle_name):
    dh_operator = data_handler.DataHandler(10000, 'pos.txt', 'neg.txt')
    dh_operator.create_pickled_data(pickle_name)

net = Network(hidden_layers = [1500, 1500, 1500], epochs = 10, batch_size = 100, pickle_name = pickle_name)
net.train()
