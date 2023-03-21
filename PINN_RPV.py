# Import Packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(124)
tf.compat.v1.set_random_seed(134)


class PINNs:
    def __init__(self, x, t, x0, t0, h0, lb, ub, layers):
        self.x = x
        self.t = t
        self.x0 = x0
        self.t0 = t0
        self.h0 = h0
        self.lb = lb    
        self.ub = ub
        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])  # Input
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])  # Input
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])  # Input
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])  # Input
        # observation points
        self.h0_pred, self.q0_pred, self.f0_pred, self.g0_pred = self.net_NS(self.x0_tf, self.t0_tf)
        # collocation points
        _, _, self.f_pred, self.g_pred = self.net_NS(self.x_tf, self.t_tf)
        self.weight=0.1
        self.loss_pde = tf.reduce_sum(tf.square(self.f0_pred) + tf.square(self.g0_pred)) +\
            tf.reduce_sum(tf.square(self.f_pred) + tf.square(self.g_pred))
        self.loss_observation = tf.reduce_sum(tf.square(self.h0 - self.h0_pred)) 
        self.loss = self.weight*self.loss_pde + self.loss_observation
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 500000,
                                                                         'maxfun': 500000,
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
        # pipe parameters
        g = 9.81
        D = 1.81
        A = 3.14 * D * D / 4
        fric = 0.012
        a = 1000
        # forward propagation
        h_and_q = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)

        # Output h and q
        h_ = h_and_q[:, 0:1]  
        q_ = h_and_q[:, 1:2] 

        # partial derivatives from the DNN
        h_t = tf.gradients(h_, t)[0]
        h_x = tf.gradients(h_, x)[0]
        q_t = tf.gradients(q_, t)[0]
        q_x = tf.gradients(q_, x)[0]

        ff = A * q_t + q_ * q_x + g * A * A * h_x+fric*q_*tf.abs(q_)/2/D
        gf = A * h_t + q_ * h_x + a * a * q_x / g
        return h_, q_, ff, gf

    def callback(self, loss):
        print('Loss: %.3e, l1: %.3f' % (loss))

    def train(self, nIter):
        self.loss_pdes=[]
        self.loss_observaitons=[]
        self.losses=[]
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, 
                    self.x0_tf: self.x0, self.t0_tf: self.t0}
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_pde_value = self.sess.run(self.loss_pde, tf_dict)
                loss_obs_value = self.sess.run(self.loss_observation, tf_dict)
                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_pdes+=[ loss_pde_value ]
                self.loss_observaitons+=[loss_obs_value ]
                self.losses+=[ loss_value ]
                print('loss_pde',loss_pde_value)
                print('loss_obs',loss_obs_value)

                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()  
            
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss])

    def predict(self, x_star, t_star):
        tf_dict = {self.x0_tf: x_star, self.t0_tf: t_star}
        h_star = self.sess.run(self.h0_pred, tf_dict)
        q_star = self.sess.run(self.q0_pred, tf_dict)
        return h_star, q_star


name = 'RPV_data'
DF1 = pd.read_excel(name + '.xlsx')
reaches = 51
length = 500
# Downsample the raw data to train
DF = DF1.iloc[::5, :]
t_space = DF['T'].to_numpy()  # time serise, array
x_space = np.linspace(0, length, reaches)  # pipe length seris, shape(reaches)
# Get pressure
p_all = np.zeros(shape=(reaches, len(DF)))
for i in range(reaches):
    p_all[i] = DF.iloc[:, 2 * i + 2]
# Get flow
q_all = np.zeros(shape=(reaches, len(DF)))
for i in range(reaches):
    q_all[i] = DF.iloc[:, 2 * i + 3]
x_all = np.tile(x_space, (len(DF), 1)).transpose()  # shape(reaches,steps)
t_all = np.tile(t_space, (reaches, 1))  # time shape(steps,steps)
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2]
N_collo = 20  # number of collocation points 
# Training Data
tlen = 199  # time points for training
idx = np.array([5,20,50],dtype='int') # index for training data
x_obs_train = np.zeros(shape=(len(idx), tlen))
t_obs_train = np.zeros(shape=(len(idx), tlen))
h_obs_train = np.zeros(shape=(len(idx), tlen))
# set training data, observation
for i in range(0, len(idx)):
    x_obs_train[i] = x_all[idx[i]][0:tlen]
    t_obs_train[i] = t_all[idx[i]][0:tlen]
    h_obs_train[i] = p_all[idx[i]][0:tlen]

# Collocation Dataset
idx_collo =  np.linspace(0,reaches-1,N_collo,dtype='int')

x_collo_train = np.zeros(shape=(len(idx_collo), tlen))
t_collo_train = np.zeros(shape=(len(idx_collo), tlen))

for i in range(0, len(idx_collo)):
    x_collo_train[i] = x_all[idx_collo[i]][0:tlen]
    t_collo_train[i] = t_all[idx_collo[i]][0:tlen]
# Define boundary limits for normalization
lb = np.array([0., 0.])
ub = np.array([length, t_space[tlen]])  # x and t?
# Flatten data for DNN
x_obs_train_flat = x_obs_train.flatten()[:, None]
t_obs_train_flat = t_obs_train.flatten()[:, None]
h_obs_train_flat = h_obs_train.flatten()[:, None]
x_all_flat=x_all.flatten()[:,None]
t_all_flat=t_all.flatten()[:,None]
x_collo_train_flat = x_collo_train.flatten()[:, None]
t_collo_train_flat = t_collo_train.flatten()[:, None]

# Build PINN
model = PINNs(x_collo_train_flat, t_collo_train_flat, x_obs_train_flat,
              t_obs_train_flat, h_obs_train_flat, lb, ub, layers)
model.train(400000)
h_pred_train, _ = model.predict(x_obs_train_flat, t_obs_train_flat)
plt.figure(1)
plt.plot(h_obs_train_flat, label='true')
plt.plot(h_pred_train, label='predicted')
plt.legend()

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
    q_test[i] = q_all[idx_test[i]][0:tlen]
x_test_flat = x_test.flatten()[:, None]
t_test_flat = t_test.flatten()[:, None]
p_test_flat = p_test.flatten()[:, None]
q_test_flat = q_test.flatten()[:, None]
h_pred, q_pred = model.predict(x_test_flat, t_test_flat)

plt.figure(2)
plt.plot(p_test_flat, label='true')
plt.plot(h_pred, label='predicted')
plt.legend()
plt.figure(3)
plt.plot(q_test_flat, label='true')
plt.plot(q_pred, label='predicted')
plt.legend()
plt.figure(4)
plt.plot(model.loss_pdes, label='pde')
plt.plot(model.loss_observaitons, label='observation')
plt.plot(model.losses, label='sum')
plt.legend()
np.save(name + 'p_test_flat2.npy', p_test_flat)
np.save(name + 'h_test_pred2.npy', h_pred)
np.save(name + 'q_test_flat2.npy', q_test_flat)
np.save(name + 'q_test_pred2.npy', q_pred)
np.save(name + 'h_obs_train_flat2.npy', h_obs_train_flat)
np.save(name + 'h_pred_train2.npy', h_pred_train)
np.save(name + 'loss_pde.npy', model.loss_pdes)
np.save(name + 'loss_obs.npy', model.loss_observaitons)
np.save(name + 'loss.npy', model.losses)
plt.show()
