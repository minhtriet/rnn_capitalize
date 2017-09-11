# x
# ↓ U
# h
# ↓ V
# o

# o_t → h_h+1
#     W

import numpy as np

H_WIDTH = 52

class RNN:

    def init(data, result):
        self.x = data
        self.y = result
        self.w_hh = np.random.rand(1, H_WIDTH)
        self.w_hy = np.random.rand(H_WIDTH, 1)

    def backprop(loss):
        dL_o = self.y[t]


    def step(x):
        h = tanh(np.dot(w_hh, h) + np.dot(w_xh, x))
        y = tanh(np.dot(w_hy, h))
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

rnn = RNN([1,2,3],[4,5,6])
rnn.run()

