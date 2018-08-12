import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class Network():

    def __init__(self, sizes):
        self.layers = []
        for i in range(1, len(sizes)):
            layer = {'weights': tf.Variable(tf.random_normal([sizes[i - 1], sizes[i]])),
                     'biases': tf.Variable(tf.random_normal([sizes[i]]))}
            self.layers.append(layer)

    def train(self, epochs, batch_size):
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        pred = self.feedforward(x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred,
                                                                         labels = y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                for _ in range(0, int(mnist.train.num_examples / batch_size)):
                    ep_x, ep_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict = {x: ep_x, y: ep_y})
                correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, float))
                print(
                    f'Accuracy: {int(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})* 100)}%, Epoch: {epoch + 1}')

    def feedforward(self, a):
        for layer in self.layers:
            a = tf.nn.tanh(tf.add(tf.matmul(a, layer['weights']), layer['biases']))
        return a


mnist = input_data.read_data_sets('/temp/data', one_hot = True)

net = Network([784, 300, 10])
net.train(epochs = 30, batch_size = 16)