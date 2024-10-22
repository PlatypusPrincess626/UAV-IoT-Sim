import numpy as np
import tensorflow as tf

import os
import sys

import pygad
import pygad.nn
import pygad.gann
import pandas as pd
import statistics as stats
from collections import deque
import random

from env_utils.logger_utils import RunningAverage

def modify_state(state):
    total_data = state[0][1]
    refined_state = [[0, 0] for _ in range(len(state))]
    np_state = np.array(state)
    _, _, zmax = np_state.max(axis=0)

    # ADF 2.0
    for i in range(len(state)):
        refined_state[i][0] = state[i][1]/max(total_data, 1)
        if i == 0:
            refined_state[i][1] = state[i][2]/6_800_000
        else:
            refined_state[i][1] = state[i][2]/max(zmax, 1)

    return refined_state

class get_ql_agent:
    def __init__(self, env,
                 epsilon_i=1.0,
                 epsilon_f=0.0,
                 n_epsilon=0.1,
                 alpha=0.5,
                 gamma=0.95):
        """
        num_states: the number of states in the discrete state space
        num_actions: the number of actions in the discrete action space
        epsilon_i: the initial value for epsilon
        epsilon_f: the final value for epsilon
        n_epsilon: a float between 0 and 1 determining at which point
        during training epsilon should have decayed from epsilon_i to
        epsilon_f
        alpha: the learning rate
        gamma: the decay rate
        """

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon = epsilon_i
        self.n_epsilon = n_epsilon

        # ADF 2.0
        self.num_actions = env.num_ch + 5
        self.num_states = 6800 + (env._num_uav + env.num_ch + 5) * (25 * env._max_steps) * (env._max_steps)
        # ADF 1.0
        # self.num_actions = env.num_ch
        # self.num_states = 6800 + (env._num_uav + env.num_ch) * (25 * env._max_steps) * (env._max_steps)
        self.encoder = np.array([[[]]])

        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((self.num_states, self.num_actions))

        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

    def encode_state(self, state):
        if self.encoder.size == 0:
            np.append(self.encoder, [state])
            return 0
        else:
            for index in range(len(self.encoder)):
                comparison = self.encoder[index] == state
                if comparison.all():
                    return index
            np.append(self.encoder, [state])
            return len(self.encoder)-1

    def decay_epsilon(self, n):
        """
        Decays the get_ddqn_agent's exploration rate according to n, which is a
        float between 0 and 1 describing how far along training is,
        with 0 meaning 'just started' and 1 meaning 'done'.
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))

    def act(self, s_t_raw):
        """
        Epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        s_t = self.encode_state(s_t_raw)
        return np.argmax(self.Q[s_t])

    def update(self, s_t_raw, a_t, r_t, s_t_next_raw, d_t):
        """
        Uses the q-learning update rule to update the get_ddqn_agent's predictions
        for Q(s_t, a_t).
        """

        s_t = self.encode_state(s_t_raw)
        s_t_next = self.encode_state(s_t_next_raw)

        Q_next = np.max(self.Q[s_t_next])

        targets = r_t + (1 - d_t) * self.gamma * Q_next

        self.qvalue_max.add(self.Q[s_t].max().item())
        self.qvalue_mean.add(self.Q[s_t].mean().item())
        self.qvalue_min.add(self.Q[s_t].min().item())

        self.target_max.add(targets.max().item())
        self.target_mean.add(targets.mean().item())
        self.target_min.add(targets.min().item())

        loss = (np.square(self.Q[s_t].max()-targets)).mean()
        self.td_errors.add(loss.item())

        self.Q[s_t, a_t] = self.Q[s_t, a_t] + self.alpha * (r_t + (1 - d_t) * self.gamma * Q_next - self.Q[s_t, a_t])

class get_gann_agent:
    def __init__(self,
                 env
                 ):
        # ADF 2.0
        self._num_inputs = 2 * (env.num_ch + 6)
        self._num_output = env.num_ch + 5
        # ADF 1.0
        # self._num_inputs = 2 * (env.num_ch + 1)
        # self._num_output = env.num_ch
        self.sol_idx = 0

        self.a_t = 0
        self.r_t = 0
        self.d_t = False
        self.gamma = 0.95
        self.last_inputs = np.expand_dims(np.array([0]*self._num_inputs).flatten(), axis=0)
        self.data_inputs = np.expand_dims(np.array([0]*self._num_inputs).flatten(), axis=0)

        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

        self.GANN_instance = pygad.gann.GANN(num_solutions=8,
                                num_neurons_input=self._num_inputs,
                                num_neurons_hidden_layers=[self._num_inputs, self._num_inputs, self._num_inputs],
                                num_neurons_output=self._num_output,
                                hidden_activations=["relu", "relu", "relu"],
                                output_activation="softmax")


    def fitness_func(self, ga_instance, solution, sol_idx):

        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[sol_idx],
                                       data_inputs=self.data_inputs)
        previous = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[sol_idx],
                         data_inputs=self.last_inputs)
        expected_error = pow(self.r_t + (1 - self.d_t) * self.gamma * np.max(predictions) - np.max(previous), 2)
        solution_fitness = (1-expected_error)*100

        return solution_fitness

    def callback_generation(self, ga_instance):
        population_matrices = pygad.gann.population_as_matrices(population_networks=self.GANN_instance.population_networks,
                                                                population_vectors=ga_instance.population)

        self.GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

        self.last_fitness = ga_instance.best_solution()[1].copy()

    def update(self, s_t, a_t, r_t, s_t_next, d_t):
        r_s_t = modify_state(s_t)
        r_s_t_next = modify_state(s_t_next)
        self.last_inputs = np.expand_dims(np.array(r_s_t).flatten(), axis=0)
        self.data_inputs = np.expand_dims(np.array(r_s_t_next).flatten(), axis=0)
        self.d_t = d_t
        self.r_t = r_t
        self.a_t = a_t

        population_vectors = pygad.gann.population_as_vectors(population_networks=self.GANN_instance.population_networks)
        initial_population = population_vectors.copy()

        ga_instance = pygad.GA(num_generations=1,
                       num_parents_mating=4,
                       initial_population=initial_population,
                       fitness_func=self.fitness_func,
                       mutation_percent_genes=10,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       keep_parents=-1,
                       on_generation=self.callback_generation)
        ga_instance.run()
        
        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[self.sol_idx],
                                       data_inputs=self.data_inputs)
        previous = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[self.sol_idx],
                                    data_inputs=self.last_inputs)
        
        self.qvalue_max.add(max(previous))
        self.qvalue_mean.add(stats.mean(previous))
        self.qvalue_min.add(min(previous))

        self.target_max.add(max(predictions))
        self.target_mean.add(stats.mean(predictions))
        self.target_min.add(min(predictions))

        loss = (np.square(max(previous) - max(predictions)))
        self.td_errors.add(loss)
        

        self.sol, self.sol_fit, self.sol_idx = ga_instance.best_solution()

    def act(self, s_t):
        mod_s_t = modify_state(s_t)
        inputs = np.expand_dims(np.array(mod_s_t).flatten(), axis=0)

        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[self.sol_idx],
                                       data_inputs=inputs)

        return np.argmax(predictions)


class get_ddqn_agent():
    def __init__(self, env, nS, nA, epsilon_i=1.0, epsilon_f=0.0, n_epsilon=0.1,
                 alpha=0.5, gamma=0.95, epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.01):
        # ADF 2.0
        self.nS = nS
        self.nA = nA
        # ADF 1.0

        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.loss = []
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.n_epsilon = n_epsilon

        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

    def build_model(self):
        model = tf.keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(tf.keras.layers.Input(shape=(self.nS, )))
        model.add(tf.keras.layers.Dense(250, activation='relu'))  # [Input] -> Layer 1
        #   Dense: Densely connected layer https://keras.io/layers/core/
        #   24: Number of neurons
        #   input_dim: Number of input variables
        #   activation: Rectified Linear Unit (relu) ranges >= 0
        model.add(tf.keras.layers.Dense(250, activation='relu'))  # Layer 2 -> 3
        model.add(tf.keras.layers.Dense(self.nA, activation='linear'))  # Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=tf.keras.optimizers.Adam(
                          learning_rate=self.alpha))  # Optimaizer: Adam (Feel free to check other options)
        return model

    def decay_epsilon(self, n):
        """
        Decays the get_ddqn_agent's exploration rate according to n, which is a
        float between 0 and 1 describing how far along training is,
        with 0 meaning 'just started' and 1 meaning 'done'.
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def act(self, state):
        r_state = modify_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        action_vals = self.model.predict(np.expand_dims(np.array(r_state).flatten(), axis=0))  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        r_state = modify_state(state)
        action_vals = self.model.predict(np.expand_dims(np.array(r_state).flatten(), axis=0))  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def update_mem(self, state, action, reward, nstate, done):
        # Store the experience in memory
        r_state = modify_state(state)
        r_nstate = modify_state(nstate)
        self.memory.append((r_state, action, reward, r_nstate, done))

    def train(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(self.memory, batch_size)  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = minibatch
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):  # Creating the state and next state np arrays
            st = np.append(st, np.expand_dims(np.array(np_array[i][0]).flatten(), axis=0), axis=0)
            nst = np.append(nst, np.expand_dims(np.array(np_array[i][3]).flatten(), axis=0), axis=0)
        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)  # Predict from the TARGET
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(np.expand_dims(np.array(state).flatten(), axis=0))
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:  # Non terminal
                target = reward + self.gamma * nst_action_predict_target[
                    np.argmax(nst_action_predict_model)]  # Using Q to get T is Double DQN

            self.qvalue_max.add(np.argmax(nst_predict))
            self.qvalue_mean.add(np.mean(nst_predict))
            self.qvalue_min.add(np.argmin(nst_predict))

            self.target_max.add(np.argmax(nst_predict_target))
            self.target_mean.add(np.mean(nst_predict_target))
            self.target_min.add(np.argmin(nst_predict_target))

            loss = (np.square(np.argmax(nst_predict_target) - np.argmax(st_predict)))
            self.td_errors.add(loss)

            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1


        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
        # Decay Epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

class get_ddqn_regression_agent():
    def __init__(self, env, nS, nA, epsilon_i=1.0, epsilon_f=0.0, n_epsilon=0.1,
                 alpha=0.5, gamma=0.95, epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.01):
        # ADF 2.0
        self.nS = nS
        self.nA = nA
        self.state2_max = env._max_steps
        self.state1_max = env.dims
        # ADF 1.0

        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.loss = []
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.n_epsilon = n_epsilon

        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

    def build_model(self):
        model = tf.keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(tf.keras.layers.Input(shape=(self.nS, )))
        model.add(tf.keras.layers.Dense(10, activation='relu'))  # [Input] -> Layer 1
        #   Dense: Densely connected layer https://keras.io/layers/core/
        #   24: Number of neurons
        #   input_dim: Number of input variables
        #   activation: Rectified Linear Unit (relu) ranges >= 0
        model.add(tf.keras.layers.Dense(10, activation='relu'))  # Layer 2 -> 3
        model.add(tf.keras.layers.Dense(self.nA, activation='sigmoid'))  # Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=tf.keras.optimizers.Adam(
                          learning_rate=self.alpha))  # Optimaizer: Adam (Feel free to check other options)
        return model

    def decay_epsilon(self, n):
        """
        Decays the get_ddqn_agent's exploration rate according to n, which is a
        float between 0 and 1 describing how far along training is,
        with 0 meaning 'just started' and 1 meaning 'done'.
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def act(self, state):
        r_state = [state[0]/self.state1_max, state[1]/6_800_000, state[2]/self.state2_max]
        print(self.epsilon)
        if random.randint(0, 1000) < self.epsilon * 1000:
            result = random.randint(0, 1000) / 1000
            print(result)
            return result

        action_vals = self.model.predict(np.reshape(np.array(r_state), (-1, self.nS)))  # Exploit: Use the NN to predict the correct action from this state
        return action_vals[0]

    def test_action(self, state):  # Exploit
        r_state = [state[0]/self.state1_max, state[1]/6_800_000, state[2]/self.state2_max]
        action_vals = self.model.predict(np.reshape(np.array(r_state), (-1, self.nS))) # Exploit: Use the NN to predict the correct action from this state
        return action_vals[0]

    def update_mem(self, state, action, reward, nstate, done):
        # Store the experience in memory
        r_state = [state[0]/self.state1_max, state[1]/6_800_000, state[2]/self.state2_max]
        r_nstate = [nstate[0]/self.state1_max, nstate[1]/6_800_000, nstate[2]/self.state2_max]
        self.memory.append((r_state, action, reward, r_nstate, done))

    def train(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(self.memory, batch_size)  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = minibatch
        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):  # Creating the state and next state np arrays
            st = np.append(st, np.reshape(np.array(np_array[i][0]), (-1, self.nS)), axis=0)
            nst = np.append(nst, np.reshape(np.array(np_array[i][3]), (-1, self.nS)), axis=0)
        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.model_target.predict(nst)  # Predict from the TARGET
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(np.reshape(np.array(state), (-1, self.nS)))
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward - self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)]  # Using Q to get T is Double DQNN
            else:  # Non terminal
                target = reward

            # self.qvalue_max.add(np.argmax(nst_predict))
            # self.qvalue_mean.add(np.mean(nst_predict))
            # self.qvalue_min.add(np.argmin(nst_predict))
            #
            # self.target_max.add(np.argmax(nst_predict_target))
            # self.target_mean.add(np.mean(nst_predict_target))
            # self.target_min.add(np.argmin(nst_predict_target))

            # loss = pow((nst_predict_target - st_predict), 2)
            # self.td_errors.add(loss)

            target_f = st_predict[index]

            target_f[0] = target
            y.append(target)
            index += 1


        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
        # Decay Epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
