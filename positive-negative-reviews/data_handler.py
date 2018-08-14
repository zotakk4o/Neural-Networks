from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter
import pickle


class DataHandler():
    def __init__(self, lines, pos, neg):
        self.lines = lines
        self.lemmatizer = WordNetLemmatizer()
        self.pos = pos
        self.neg = neg
        self.lexicon = []

    def create_lexicon(self):
        for file in [self.pos, self.neg]:
            with open(file, 'r') as f:
                file_lines = f.readlines()
                for line in file_lines[:self.lines]:
                    tokenized_line = word_tokenize(line)
                    self.lexicon += list(self.lemmatizer.lemmatize(w) for w in tokenized_line)

        words_counter = Counter(self.lexicon)
        output = []

        for word in words_counter:
            if 1000 > words_counter[word] > 50:
                output.append(word)
        return output

    def classify_input(self, sample, lexicon, classification):
        output = []

        with open(sample, 'r') as f:
            content = f.readlines()
            for line in content[:self.lines]:
                line = word_tokenize(line)
                line = [self.lemmatizer.lemmatize(word) for word in line]
                classified_input = np.zeros(len(lexicon))
                for word in line:
                    word = word.lower()
                    if word in lexicon:
                        index = lexicon.index(word)
                        classified_input[index] += 1

                output.append([classified_input, classification])

        return output

    def create_test_and_train_data(self, test_size = 10):

        lexicon = self.create_lexicon()
        input_samples = []
        input_samples += self.classify_input(self.pos, lexicon, [1, 0])
        input_samples += self.classify_input(self.neg, lexicon, [0, 1])
        np.random.shuffle(input_samples)

        test_size = int((test_size / 100) * len(input_samples))
        input_samples = np.array(input_samples)

        train_x = list(input_samples[:, 0][:-test_size])
        train_y = list(input_samples[:, 1][:-test_size])
        test_x = list(input_samples[:, 0][-test_size:])
        test_y = list(input_samples[:, 1][-test_size:])

        return train_x, train_y, test_x, test_y

    def create_pickled_data(self, file_path_and_name):
        train_x, train_y, test_x, test_y = self.create_test_and_train_data()
        with open(file_path_and_name, 'wb') as f:
            pickle.dump([train_x, train_y, test_x, test_y], f)
