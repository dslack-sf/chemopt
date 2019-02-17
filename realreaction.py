import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
import json

import os

import rnn
from reactions import QuadraticEval, ConstraintQuadraticEval, RealReaction
from logger import get_handlers
from collections import namedtuple
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd

NUM_DIMENSIONS = 3

logging.basicConfig(level=logging.INFO, handlers=get_handlers())
logger = logging.getLogger()

state_space = pd.read_csv('EtNH3Istateset.csv')

class StepOptimizer:
    def __init__(self, cell, func, ndim, nsteps, save_path,ckpt_path, logger, constraints):
        self.logger = logger
        self.cell = cell
        self.func = func
        self.ndim = ndim
        self.save_path = save_path
        self.nsteps = nsteps
        self.ckpt_path = ckpt_path
        self.constraints = constraints
        self.init_state = self.cell.get_initial_state(1, tf.float32)
        self.results = self.build_graph()
        self.saver = tf.train.Saver(tf.global_variables())
        self.next_idx = None

    def get_state_shapes(self):
        return [(s[0].get_shape().as_list(), s[1].get_shape().as_list())
                for s in self.init_state]

    def step(self, sess, x, y, state):
        feed_dict = {'input_x:0':x, 'input_y:0':y}
        for i in range(len(self.init_state)):
            feed_dict['state_l{0}_c:0'.format(i)] = state[i][0]
            feed_dict['state_l{0}_h:0'.format(i)] = state[i][1]

        new_x, new_state = sess.run(self.results, feed_dict=feed_dict)
  
        readable_x = self.func.x_convert(np.squeeze(new_x)).reshape(1, NUM_DIMENSIONS)
        distances = euclidean_distances(state_space, readable_x)
        distances = list(distances)
        min_distance = min(distances)
        indx = distances.index(min_distance)

        # Set next index for extraction
        self.next_idx = indx

        new_x = self.func.x_normalize(state_space.loc[indx])
        new_x = new_x.reshape(1,3)

        return new_x, new_state

    def build_graph(self):
        x = tf.placeholder(tf.float32, shape=[1, self.ndim], name='input_x')
        y = tf.placeholder(tf.float32, shape=[1, 1], name='input_y')
        state = []
        for i in range(len(self.init_state)):
            state.append((tf.placeholder(
                              tf.float32, shape=self.init_state[i][0].get_shape(),
                              name='state_l{0}_c'.format(i)),
                          tf.placeholder(
                              tf.float32, shape=self.init_state[i][1].get_shape(),
                              name='state_l{0}_h'.format(i))))

        with tf.name_scope('opt_cell'):
            new_x, new_state = self.cell(x, y, state)
            if self.constraints:
                new_x = tf.clip_by_value(new_x, 0.01, 0.99)
        return new_x, new_state

    def load(self, sess, ckpt_path):
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('Reading model parameters from {}.'.format(
                ckpt.model_checkpoint_path))
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('No checkpoint available')

    def save(self, sess, save_path):
        self.saver.save(sess, save_path)

    def get_init(self, starting_location):    
        x = self.func.x_normalize(list(state_space.loc[starting_location]))
        x = x.reshape(1,3)
        y = np.array(self.func(x)).reshape(1, 1)

        init_state = [(np.zeros(s[0]), np.zeros(s[1]))
                      for s in self.get_state_shapes()]
        return x, y, init_state

    def run(self, previous_location):
        with tf.Session() as sess:
            self.load(sess, self.ckpt_path)
            x, y, state = self.get_init(previous_location)

            x_array = np.zeros((self.nsteps + 1, self.ndim))
            y_array = np.zeros((self.nsteps + 1, 1))
            x_array[0, :] = x
            y_array[0] = y
            # for i in range(self.nsteps):
            #     x, state = self.step(sess, x, y, state)
            #     y = np.array(self.func(x)).reshape(1, 1)
            #     x_array[i+1, :] = x
            #     y_array[i+1] = y

            self.save(sess, self.save_path)
        return x_array, y_array

    def run_current_step(self):
        """
        * Using most recent experiment, get next step
        * Save across the other steps
        """
        pass

def main():
    config_file = open('./config.json')
    config = json.load(config_file,
                       object_hook=lambda d:namedtuple('x', d.keys())(*d.values()))

    # update number of parameters to all those considred in the data set
    param_names = []
    param_range = []
    for col in state_space:
        param_names.append(col)
        param_range.append((state_space[col].min(), state_space[col].max()))

    func = RealReaction(num_dim = len(param_names), param_range=param_range, param_names=param_names,
                        direction='max', logger=None)

    cell = rnn.StochasticRNNCell(cell=rnn.LSTM,
                                 kwargs={'hidden_size':config.hidden_size},
                                 nlayers=config.num_layers,
                                 reuse=config.reuse)
    # Assumes that previous step exists
    # e.g. current_step = 2 means that the first step is in place
    next_states = []
    for baseline_num in range(1, 10):
        # print (config.sav)
        df = pd.read_csv('./ckpt/baseline/baseline_{}/trace.csv'.format(baseline_num))
        l = list(df['step'])
        current_step = len(l) + 1 
        save_path_of_previous_step = "./ckpt/baseline/baseline_{}/step{}".format(baseline_num, current_step - 1)
        save_path = './ckpt/baseline/baseline_{}/step{}'.format(baseline_num, current_step)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        optimizer = StepOptimizer(cell=cell, func=func, ndim=config.num_params,
                                  nsteps=1, save_path=save_path,
                                  ckpt_path=save_path_of_previous_step,logger=logger,
                                  constraints=config.constraints)

        print (save_path_of_previous_step)
        x_array, y_array = optimizer.run(l[-1])

        l.append(optimizer.next_idx)
        pd.DataFrame(l).to_csv('./ckpt/baseline/baseline_{}/trace.csv'.format(baseline_num))
        next_states.append(optimizer.next_idx)


    pritn (next_states)
        
    # plt.figure(1)
    # plt.plot(y_array)
    # plt.show()
    
if __name__ == '__main__':
    main()
