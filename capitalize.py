#  x
#  |
# \|/ U
#  h
#  |
# \|/ V
#  o

# o_t -> h_h+1
#     W

import numpy as np
import pdb
import string

VOCAB_SIZE = 26
HIDDEN_NEURON_SIZE = 20
UNROLL_LENGTH = 50
CLASSES = 11
ETA = 0.0001

def softmax(x):
    e_x = np.exp(x - max(x))
    return x / e_x.sum()

class RNN:

    def __init__(self):
        self.w_xh = np.random.rand(VOCAB_SIZE, HIDDEN_NEURON_SIZE) * 0.01
        self.w_hh = np.random.rand(HIDDEN_NEURON_SIZE, HIDDEN_NEURON_SIZE) * 0.01
        self.w_hy = np.random.rand(HIDDEN_NEURON_SIZE, CLASSES) * 0.01  # TODO but why HIDDEN_NEURON_SIZE
        self.b_h = np.random.rand(1, HIDDEN_NEURON_SIZE) * 0.01
        self.b_y = np.random.rand(1, CLASSES) * 0.01
        self.loss = 0

    def step(self, data, target):
        h[-1] = np.zeros([1, HIDDEN_NEURON_SIZE])
        for i in reversed(xrange(len(data))):
            x[i] = np.zeros([1, VOCAB_SIZE])  # onehot
            x[0][[char_to_ix[ data[i] ]]] = 1
            h[i] = np.tanh(np.dot(h[i-1], self.w_hh) + np.dot(x[i], self.w_xh) + self.b_h)
            y = np.dot(h[i], self.w_hy) + self.b_y
            predict = softmax(y)
            self.loss += -np.log(predict[0, target])
        # backward
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by)
        dhnext = np.zeros_like(h[0])
        for i in reversed(xrange(len(data))):
            # backward
            dy = np.copy(predict)
            dy[ target ] -= 1       # softmax derivative
            dWhy = np.dot(self.w_hy.T, dy))
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext
            dhraw = (1 - np.square(h[i])) * dh
            dbh += dhraw
            dWxh = 1 - np.square(h[i]) * x[i]
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
        self.w_xh += ETA*dWxh
        self.w_hh += ETA*dWhh
        self.w_hy += ETA*dWhy
        self.b_h += ETA*dbh
        self.b_y += ETA*dby
        print dWxh, dWhh, dWhy
        return dWxh, dWhh, dWhy

data = open('input.txt', 'r').readlines() # should be simple plain text file
result = open('result.txt', 'r').readlines()
char_to_ix = dict(zip(string.ascii_lowercase, range(0,26)))
rnn = RNN()
for index_i, sentence in enumerate(data):              # TODO why while true?
    print "Step %d" % index_i
    rnn.step(sentence.strip(), int(result[index_i]))
