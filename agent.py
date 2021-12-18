from re import S
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import convert_to_tensor as ctt
from tensorflow import GradientTape
from tensorflow import float32 as float32
from tensorflow.keras.optimizers import Adam
from buffer_memory import ReplayBuffer
from nw import CriticNetwork, ActorNetwork


class Agent:
    def __init__(self, input_dims, alpha=.001, beta=0.002, env=None,
                 gamma=0.99,
                 n_actions=2,
                 max_size=100000, tau=.005, fc1=400, fc2=300, batch_size=64, noise=.1):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.noise = noise
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name="actor")
        self.critic = CriticNetwork(name="critic")  # critic network

        self.target_actor = ActorNetwork(
            n_actions=n_actions,
            name="target_actor")  # target critic network

        self.target_critic = CriticNetwork(name="target_critic")

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.get_weights()

        for i, weight in enumerate(self.actor.get_weights()):
            w = weight*tau+targets[i]*(1-tau)
            weights.append(w)
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.get_weights()
        for i, weight in enumerate(self.critic.get_weights()):
            w = weight*tau+targets[i]*(1-tau)
            weights.append(w)
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("Saving models to: ", self.actor.checkpoint_file)
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print("Loading models from: ", self.actor.checkpoint_file)
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = ctt([observation], dtype=float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0, stddev=self.noise)

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)

        states = ctt(state, dtype=float32)
        new_states = ctt(new_state, dtype=float32)
        actions = ctt(action, dtype=float32)
        rewards = ctt(reward, dtype=float32)

        with GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            critic_value_ = tf.squeeze(self.target_critic(
                new_states, target_actions), 1)

            critic_value = tf.squeeze(self.critic(states, actions), 1)

            target = rewards + self.gamma*critic_value_*(1-done)

            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradients = tape.gradient(
            critic_loss, self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(
            zip(critic_network_gradients, self.critic.trainable_variables))

        with GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.reduce_mean(actor_loss)

        actor_network_gradients = tape.gradient(actor_loss,
                                                self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_network_gradients,
                                                 self.actor.trainable_variables))

        self.update_network_parameters()
