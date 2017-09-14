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

VOCAB_SIZE = 52
HIDDEN_NEURON_SIZE = 20
UNROLL_LENGTH = 50
CLASSES = 4
class RNN:

    def __init__(self):
        self.w_xh = np.random.rand(HIDDEN_NEURON_SIZE, VOCAB_SIZE) 
        self.w_hh = np.random.rand(HIDDEN_NEURON_SIZE, HIDDEN_NEURON_SIZE)
        self.w_hy = np.random.rand(VOCAB_SIZE, HIDDEN_NEURON_SIZE)  # TODO but why HIDDEN_NEURON_SIZE
        self.b_hh = np.random.rand(HIDDEN_NEURON_SIZE, 1)
        self.b_hy = np.random.rand(VOCAB_SIZE, 1)
        self.loss = 0


    def step(x, target):
        h = np.tanh(np.dot(self.w_hh, h) + np.dot(self.w_xh, x))
        y = np.dot(self.w_hy, h)
        predict = softmax(y)
        loss = -np.log(target * predict)
        
        # backward
        dL_o = 1 - predict 
        return loss 

    def softmax(x):
        e_x = np.exp(x - max(x))
        return x / e_x.sum()


data = open('input.txt', 'r').read() # should be simple plain text file
result = open('result.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

rnn = RNN()
for i in data:              # TODO why while true?
    onehot_encode = np.zeros([len(i), VOCAB_SIZE])
    for index, j in enumerate(i):
        onehot_encode[index][ char_to_ix[j] ] = 1
    loss = rnn.step(onehot_encode)

