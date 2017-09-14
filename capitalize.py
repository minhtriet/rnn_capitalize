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

VOCAB_SIZE = 26 
HIDDEN_NEURON_SIZE = 20
UNROLL_LENGTH = 50
CLASSES = 4
class RNN:

    def __init__(self):
        self.w_xh = np.random.rand(HIDDEN_NEURON_SIZE, VOCAB_SIZE) 
        self.w_hh = np.random.rand(HIDDEN_NEURON_SIZE, HIDDEN_NEURON_SIZE)
        self.w_hy = np.random.rand(HIDDEN_NEURON_SIZE, CLASSES)  # TODO but why HIDDEN_NEURON_SIZE
        self.b_hh = np.random.rand(HIDDEN_NEURON_SIZE, 1)
        self.b_hy = np.random.rand(VOCAB_SIZE, 1)
        self.h = np.zeros([VOCAB_SIZE, HIDDEN_NEURON_SIZE])
        self.loss = 0

    def step(self, x, target):
        self.h = np.tanh(np.dot(self.h, self.w_hh) + np.dot(x, self.w_xh))
        y = np.dot(self.h, self.w_hy)
        predict = softmax(y)
        loss = -np.log(target * predict)
        # backward
        dL_o = 1 - predict 
        return loss 

    def softmax(x):
        e_x = np.exp(x - max(x))
        return x / e_x.sum()


data = open('input.txt', 'r').readlines() # should be simple plain text file
result = open('result.txt', 'r').readlines()
char_to_ix = dict(zip(string.ascii_lowercase, range(1,27))) 
rnn = RNN()
for index_i, sentence in enumerate(data):              # TODO why while true?
    print "Step %d" % index_i
    sentence = sentence.strip()
    onehot_encode = np.zeros([len(sentence), VOCAB_SIZE])
    pdb.set_trace() 
    for index, j in enumerate(sentence):
        onehot_encode[index][ char_to_ix[j] ] = 1
    loss = rnn.step(onehot_encode, result[index_i])

