# x
# ↓ U
# h
# ↓ V
# o

# o_t → h_h+1
#     W

import numpy as np

VOCAB_SIZE = 52
HIDDEN_NEURON_SIZE = 20
UNROLL_LENGTH = 50

class RNN:

    def __init__(self):
        self.w_xh = np.random.rand(HIDDEN_NEURON_SIZE, VOCAB_SIZE) 
        self.w_hh = np.random.rand(HIDDEN_NEURON_SIZE, HIDDEN_NEURON_SIZE)
        self.w_hy = np.random.rand(VOCAB_SIZE, HIDDEN_NEURON_SIZE)  # TODO but why HIDDEN_NEURON_SIZE
        self.b_hh = np.random.rand(HIDDEN_NEURON_SIZE, 1)
        self.b_hy = np.random.rand(VOCAB_SIZE, 1)

    def backprop(loss):
        dL_o = self.y[t]


    def step(x):
        h = tanh(np.dot(self.w_hh, h) + np.dot(self.w_xh, x))
        y = tanh(np.dot(self.w_hy, h))
        return y

    def softmax(x):
        e_x = np.exp(x - max(x))
        return x / e_x.sum()

    def compute_loss(predict):
        return np.log(softmax(x))

    def run():
        while True:
            predict = step(self.x)
            loss = compute_loss(predict)
            if loss < exp(-6):
                return
            back_prop(loss)

data = open('input.txt', 'r').read() # should be simple plain text file
result = open('result.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

rnn = RNN()
rnn.run()

