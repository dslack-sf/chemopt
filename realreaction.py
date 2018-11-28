import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
import json

import rnn
from reactions import QuadraticEval, ConstraintQuadraticEval, RealReaction
from logger import get_handlers
from collections import namedtuple

import pandas as pd

NUM_DIMENSIONS = 82

logging.basicConfig(level=logging.INFO, handlers=get_handlers())
logger = logging.getLogger()

start_df = pd.read_csv('dylan/Center_Dataframe.csv')
start_location = start_df.loc[[0]].as_matrix()
start_location = np.delete(start_location, 0, 1)

class StepOptimizer:
    def __init__(self, cell, func, ndim, nsteps, ckpt_path, logger, constraints):
        self.logger = logger
        self.cell = cell
        self.func = func
        self.ndim = ndim
        self.nsteps = nsteps
        self.ckpt_path = ckpt_path
        self.constraints = constraints
        self.init_state = self.cell.get_initial_state(1, tf.float32)
        self.results = self.build_graph()

        self.saver = tf.train.Saver(tf.global_variables())

    def get_state_shapes(self):
        return [(s[0].get_shape().as_list(), s[1].get_shape().as_list())
                for s in self.init_state]

    def step(self, sess, x, y, state):
        feed_dict = {'input_x:0':x, 'input_y:0':y}
        for i in range(len(self.init_state)):
            feed_dict['state_l{0}_c:0'.format(i)] = state[i][0]
            feed_dict['state_l{0}_h:0'.format(i)] = state[i][1]
        new_x, new_state = sess.run(self.results, feed_dict=feed_dict)
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

    def get_init(self):

        ### Init is here!
        # Want to choose correct starting location everytime
        # if start_location:
        x = np.random.normal(loc=0.5, scale=0.2, size=(1, NUM_DIMENSIONS))
        x = np.maximum(np.minimum(x, 0.9), 0.1)
        
        x = start_location


        y = np.array(self.func(x)).reshape(1, 1)
        init_state = [(np.zeros(s[0]), np.zeros(s[1]))
                      for s in self.get_state_shapes()]
        return x, y, init_state

    def run(self):
        with tf.Session() as sess:
            self.load(sess, self.ckpt_path)
            x, y, state = self.get_init()
            x_array = np.zeros((self.nsteps + 1, self.ndim))
            y_array = np.zeros((self.nsteps + 1, 1))
            x_array[0, :] = x
            y_array[0] = y
            for i in range(self.nsteps):
                x, state = self.step(sess, x, y, state)
                y = np.array(self.func(x)).reshape(1, 1)
                x_array[i+1, :] = x
                y_array[i+1] = y

        return x_array, y_array

def main():
    config_file = open('./config.json')
    config = json.load(config_file,
                       object_hook=lambda d:namedtuple('x', d.keys())(*d.values()))


    ## Load data, seperate labels
    preloaded_data_from_cvs = pd.read_csv('trainingset.csv')
    preloaded_data_from_cvs = preloaded_data_from_cvs.loc[preloaded_data_from_cvs['_rxn_organic-inchikey'] == "UPHCENSIMPJEIS-UHFFFAOYSA-N"]
    preloaded_data_from_cvs = preloaded_data_from_cvs.sample(frac = 1).reset_index(drop=True)
    
    labels = pd.DataFrame()
    labels['labels'] = preloaded_data_from_cvs['_out_crystalscore']
    preloaded_data_from_cvs = preloaded_data_from_cvs.drop(['RunID_vial', '_out_crystalscore', '_rxn_organic-inchikey'], axis = 1)
    new_labels = []

    # Adjust here if you want to change labels to 0 - 1 scaled values
    for val in labels['labels']: 
        new_labels.append(val/4.0)

    labels = labels.drop('labels', axis = 1)
    labels['labels'] = new_labels

    # update number of parameters to all those considred in the data set
    param_names = []
    param_range = []
    for col in preloaded_data_from_cvs:
        param_names.append(col)
        param_range.append((preloaded_data_from_cvs[col].min(), preloaded_data_from_cvs[col].max()))

    func = RealReaction(num_dim = len(param_names), param_range=param_range, param_names=param_names,
                        direction='max', logger=None, discrete_data_points = preloaded_data_from_cvs,
                        discrete_yields = labels)

    cell = rnn.StochasticRNNCell(cell=rnn.LSTM,
                                 kwargs={'hidden_size':config.hidden_size},
                                 nlayers=config.num_layers,
                                 reuse=config.reuse)

    optimizer = StepOptimizer(cell=cell, func=func, ndim=config.num_params,
                              nsteps=config.num_steps,
                              ckpt_path=config.save_path, logger=logger,
                              constraints=config.constraints)
    x_array, y_array = optimizer.run()

    
    plt.figure(1)
    plt.plot(y_array)
    plt.show()
    
if __name__ == '__main__':
    main()
