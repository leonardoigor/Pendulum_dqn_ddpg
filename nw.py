import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import os


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
                 name='critic', chkpt_dir='./checkpoints/'):

        super(CriticNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(1)
        self.chkpt_dir = chkpt_dir

        self.model_name = name
        self.checkpoint_dit = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dit, self.model_name+'_dddpg.h5')

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q_value = self.q(action_value)

        return q_value


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor',
                 chkpt_dir='./checkpoints'):

        super(ActorNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.n_actions = n_actions

        self.model_name = name

        self.checkpoint_dit = chkpt_dir
        self.checkpoint_file = os.path.join(
            self.checkpoint_dit, self.model_name+'_dddpg.h5')

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.mu = Dense(n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        return mu
