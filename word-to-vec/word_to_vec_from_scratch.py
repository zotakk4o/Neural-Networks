import time
import numpy as np
import operator
import math
import re


class Word_To_Vec_FS():

    def __init__(self, file_name, window_size = 2):
        self.txt_name = file_name
        self.window_size = window_size
        self.vocab_len = 0
        self.eta = 0.01

        self.word2index = {}
        self.index2word = {}
        self.weights = []
        self.biases = []
        self.layers = []
        self.vocab = []

        self.convert_text_to_vec()

    def convert_text_to_vec(self):
        with open(self.txt_name, 'r') as f:
            f_data = f.readlines()
            index = 0
            print('Creating Vocabulary...')
            for line in f_data:
                line = line.split()
                for l in line:
                    match = re.match("[^-][a-zA-Z0-9-]+", l)
                    if match:
                        match = match.group(0).lower()
                        self.vocab.append(match)
                        if match not in self.word2index:
                            self.word2index[match] = index
                            self.index2word[index] = match
                            index += 1

            self.vocab_len = len(self.word2index)
            self.initialize_training_parameters()

            print(f"Vocabulary, biases and weights created for {self.vocab_len} words")

    def initialize_training_parameters(self):
        self.layers = [self.vocab_len, 100, self.vocab_len]
        self.weights = [np.random.uniform(-0.8, 0.8, (x, y)) for x, y in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.uniform(-0.8, 0.8, (x, 1)) for x in self.layers[1:]]

    def one_hot_encode_vec(self, word):
        output_arr = np.zeros(self.vocab_len)
        output_arr[self.word2index[word]] = 1
        return output_arr

    def skip_gram(self):
        for index, word in enumerate(self.vocab):
            input_vec = self.one_hot_encode_vec(word)
            output_vec = []

            for prev in range(index - 1, max(index - 1 - int(self.window_size / 2), -1), -1):
                output_vec.append(self.one_hot_encode_vec(self.vocab[prev]))

            for following in range(index + 1, min(index + 1 + int(self.window_size / 2), self.vocab_len)):
                output_vec.append(self.one_hot_encode_vec(self.vocab[following]))

            yield np.array([input_vec, output_vec])

    def train_word_to_vec(self, epochs):
        print('Training...')
        progress = 0
        last_progress = 0
        progress_time = time.time()
        for epoch in range(epochs):
            if progress > last_progress + 5:
                print(f"Progress: {math.floor(progress)}%, Time per 5%: {format(time.time() - progress_time, '.2f')} seconds")
                last_progress = progress
                progress_time = time.time()

            progress = epoch * 100 / epochs
            for encoded_input in self.skip_gram():
                output_softmax, hidden, output = self.feed_forward(encoded_input[0])
                y = encoded_input[1]
                error = np.sum([np.subtract(output_softmax, word) for word in y], axis = 0)
                self.backprop(error, hidden, encoded_input[0])
        print('Training has finished successfully.')

    def feed_forward(self, x):
        hidden_layer = np.dot(self.weights[0].T, x)
        output_layer = np.dot(self.weights[1].T, hidden_layer)

        return self.stable_softmax(output_layer), hidden_layer, output_layer

    def backprop(self, error, hidden, word):
        dl_dw2 = np.outer(hidden, error)
        dl_dw1 = np.outer(word, np.dot(self.weights[1], error.T))

        self.weights[0] = self.weights[0] - (self.eta * dl_dw1)
        self.weights[1] = self.weights[1] - (self.eta * dl_dw2)
        pass

    def stable_softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis = 0)

    def process_word(self, word):
        if word not in self.vocab:
            print('Word not found in vocabulary!')

        else:
            print(f"Target word: {word}")
            dict = {}
            output_softmax, _, _ = self.feed_forward(self.one_hot_encode_vec(word))
            for index, probability in enumerate(output_softmax):
                dict[self.index2word[index]] = probability

            sorted_dict = sorted(dict.items(), key = operator.itemgetter(1))
            for key, value in sorted_dict:
                value = format(value, '.5f')
                print(f"Context word: {key}, Probability: {value}")


w2v = Word_To_Vec_FS('dataset.txt', 5)
w2v.train_word_to_vec(100)
w2v.process_word('quick')
print('-------------')
w2v.process_word('fox')
print('-------------')
w2v.process_word('dog')
