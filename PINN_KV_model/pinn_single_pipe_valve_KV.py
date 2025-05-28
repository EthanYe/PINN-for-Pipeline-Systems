# Import Packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
# tf.random.set_seed(1234)
# tf.set_random_seed(1234)
tf.compat.v1.set_random_seed(1234)


class PINNs:
    def __init__(self, x, t, x0, t0, h0, lb, ub, layers):

        X = np.concatenate([x, t], 1)  # x,t bound
        self.lb = lb
        self.ub = ub

        self.X = X

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]

        self.x0 = x0
        self.t0 = t0
        self.h0 = h0

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable(tf.constant(0.1), dtype=tf.float32)
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])  # Input
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])  # Input

        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])  # Input
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])  # Input
        self.h0_tf = tf.placeholder(tf.float32, shape=[None, self.h0.shape[1]])  # Output

        self.h0_pred, self.q0_pred, self.f0_pred, self.g0_pred = self.net_NS(self.x0_tf, self.t0_tf)
        _, _, self.f_pred, self.g_pred = self.net_NS(self.x_tf, self.t_tf)

        self.loss = 100*tf.reduce_sum(tf.square(self.h0_tf - self.h0_pred)) + \
            tf.reduce_sum(tf.square(self.f0_pred) + tf.square(self.g0_pred)) +\
            tf.reduce_sum(tf.square(self.f_pred) + tf.square(self.g_pred))
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        # forward propagation, from x to Y
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # update H
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, t):
        g = 9.81
        D = 1.81
        A = 3.14 * D * D / 4
        fric = 0.012
        a = 1000
        lambda_1 = self.lambda_1

        h_and_q = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        h_ = h_and_q[:, 0:1]  # Output h
        q_ = h_and_q[:, 1:2]  # Output q

        h_t = tf.gradients(h_, t)[0]
        h_x = tf.gradients(h_, x)[0]

        q_t = tf.gradients(q_, t)[0]
        q_x = tf.gradients(q_, x)[0]

        ff = A * q_t + q_ * q_x + g * A * A * h_x
        gf = A * h_t + q_ * h_x + a * a * q_x / g

        return h_, q_, ff, gf


    def callback(self, loss, lambda_1):
        print('Loss: %.3e, l1: %.3f' % (loss, lambda_1))

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.x0_tf: self.x0,
                   self.t0_tf: self.t0, self.h0_tf: self.h0}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                print('It: %d, Loss: %.3e, l1: %.3f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1],
                                loss_callback=self.callback)

    def predict(self, x_star, t_star):
        tf_dict = {self.x0_tf: x_star, self.t0_tf: t_star}
        h_star = self.sess.run(self.h0_pred, tf_dict)
        q_star = self.sess.run(self.q0_pred, tf_dict)
        return h_star, q_star

name = 'kv_data'
DF1 = pd.read_excel(name + '.xlsx')
reaches = 51
length = 500
# Take every 5th rows in original data to train
DF = DF1.iloc[::5, :]
t_all_len = DF['T'].to_numpy().reshape(len(DF['T'].to_numpy()), 1)  # get time series, column
t_space = DF['T'].to_numpy()  # time serise, array
x_space = np.linspace(0, length, reaches)  # pipe length seris, shape(reaches)
# Get pressure
p_all = np.zeros(shape=(reaches, len(DF)))
for i in range(reaches):
    p_all[i] = DF.iloc[:, 2 * i + 2]

x_all = np.tile(x_space, (len(t_all_len), 1)).transpose()  # shape(reaches*steps)
t_all = np.tile(t_space, (len(t_all_len), 1))  # time shape(steps*steps)

layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2]

N_collo = 20  # to compute ff,gf at these points
tlen = 199  # time length for training
idx = np.array([10,  30, 40],dtype='int')
x_obs_train = np.zeros(shape=(len(idx), tlen))
t_obs_train = np.zeros(shape=(len(idx), tlen))
h_obs_train = np.zeros(shape=(len(idx), tlen))
# set training data, observation
for i in range(0, len(idx)):
    x_obs_train[i] = x_all[idx[i]][0:tlen]
    t_obs_train[i] = t_all[idx[i]][0:tlen]
    h_obs_train[i] = p_all[idx[i]][0:tlen]

# Bound Data,  including x,t,p
collo_points =  np.linspace(0,reaches-1,N_collo,dtype='int')

x_bound_train = np.zeros(shape=(len(collo_points), tlen))
t_bound_train = np.zeros(shape=(len(collo_points), tlen))

for i in range(0, len(collo_points)):
    x_bound_train[i] = x_all[collo_points[i]][0:tlen]
    t_bound_train[i] = t_all[collo_points[i]][0:tlen]
# Define boundary
lb = np.array([0., 0.])

ub = np.array([length, t_space[tlen]])  # x and t?
# Flatten data for DNN
x_obs_train_flat = x_obs_train.flatten()[:, None]
t_obs_train_flat = t_obs_train.flatten()[:, None]
h_obs_train_flat = h_obs_train.flatten()[:, None]

x_bound_train_flat = x_bound_train.flatten()[:, None]
t_bound_train_flat = t_bound_train.flatten()[:, None]



model = PINNs(x_bound_train_flat, t_bound_train_flat, x_obs_train_flat,
              t_obs_train_flat, h_obs_train_flat, lb, ub, layers)
model.train(300000)
h_pred_train, _ = model.predict(x_obs_train_flat, t_obs_train_flat)
plt.figure(1)
plt.plot(h_obs_train_flat, label='true')
plt.plot(h_pred_train, label='predicted')
plt.legend()
# plt.ylim(97,103)
# Test Data
Ntest = 10
tlen = 200
idx_test = np.arange(0, 51, 5)


x_test = np.zeros(shape=(len(idx_test), tlen))
t_test = np.zeros(shape=(len(idx_test), tlen))
p_test = np.zeros(shape=(len(idx_test), tlen))
q_test = np.zeros(shape=(len(idx_test), tlen))
for i in range(0, len(idx_test)):
    x_test[i] = x_all[idx_test[i]][0:tlen]
    t_test[i] = t_all[idx_test[i]][0:tlen]
    p_test[i] = p_all[idx_test[i]][0:tlen]
x_test_flat = x_test.flatten()[:, None]
t_test_flat = t_test.flatten()[:, None]
p_test_flat = p_test.flatten()[:, None]
h_pred, q_pred = model.predict(x_test_flat, t_test_flat)

plt.figure(2)
plt.plot(p_test_flat, label='true')
plt.plot(h_pred, label='predicted')
plt.legend()
plt.show()

