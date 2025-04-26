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
    refined_state = [[0, 0, 0] for _ in range(len(state))]
    np_state = np.array(state)
    _, _, zmax, _ = np_state.max(axis=0)

    # ADF 2.0
    for i in range(len(state)):
        refined_state[i][0] = state[i][1] / max(total_data, 1)
        if i == 0:
            refined_state[i][1] = state[i][2] / 6_800_000
        else:
            refined_state[i][1] = state[i][2] / max(zmax, 1)
            refined_state[i][2] = state[i][3] / max(zmax, 1)

    return refined_state


class get_ql_agent:
    def __init__(self, env, nS, nA,
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
        self.num_actions = nA
        self.num_states = nS
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
            return len(self.encoder) - 1

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

    def update(self, s_t_raw, a_t, r_t, s_t_next_raw, d_t, step):
        """
        Uses the q-learning update rule to update the get_ddqn_agent's predictions
        for Q(s_t, a_t).
        """

        s_t = self.encode_state(s_t_raw)
        s_t_next = self.encode_state(s_t_next_raw)

        Q_next = np.max(self.Q[s_t_next])

        targets = (np.array([0.6, 0.3, 0.1]) @ r_t) + (1 - d_t) * self.gamma * Q_next

        # self.qvalue_max.add(max(a_t))
        # self.qvalue_mean.add(stats.mean(a_t))
        # self.qvalue_min.add(min(a_t))
        #
        # self.target_max.add(max(targets))
        # self.target_mean.add(stats.mean(targets))
        # self.target_min.add(min(targets))

        # loss = (np.square(self.Q[s_t].max()-targets)).mean()
        # self.td_errors.add(loss)

        self.Q[s_t, a_t] = (self.Q[s_t, a_t] + self.alpha *
                            ((np.array([0.6, 0.3, 0.1]) @ r_t) + (1 - d_t) * self.gamma * Q_next - self.Q[s_t, a_t]))


class get_gann_agent:
    def __init__(self,
                 env, nS, nA,
                 ):
        # ADF 2.0
        self._num_inputs = nS
        self._num_output = nA
        # ADF 1.0
        # self._num_inputs = 2 * (env.num_ch + 1)
        # self._num_output = env.num_ch
        self.sol_idx = 0

        self.a_t = 0
        self.r_t = 0
        self.d_t = False
        self.gamma = 0.95
        self.last_inputs = np.expand_dims(np.array([0] * self._num_inputs).flatten(), axis=0)
        self.data_inputs = np.expand_dims(np.array([0] * self._num_inputs).flatten(), axis=0)

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
                                             num_neurons_hidden_layers=[self._num_inputs, self._num_inputs,
                                                                        self._num_inputs],
                                             num_neurons_output=self._num_output,
                                             hidden_activations=["relu", "relu", "relu"],
                                             output_activation="softmax")

    def fitness_func(self, ga_instance, solution, sol_idx):
        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[sol_idx],
                                       data_inputs=self.data_inputs)
        previous = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[sol_idx],
                                    data_inputs=self.last_inputs)
        expected_error = pow(self.r_t + (1 - self.d_t) * self.gamma * np.max(predictions) - np.max(previous), 2)
        solution_fitness = (1 - expected_error) * 100

        return solution_fitness

    def callback_generation(self, ga_instance):
        population_matrices = pygad.gann.population_as_matrices(
            population_networks=self.GANN_instance.population_networks,
            population_vectors=ga_instance.population)

        self.GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

        self.last_fitness = ga_instance.best_solution()[1].copy()

    def update(self, s_t, a_t, r_t, s_t_next, d_t, step):
        r_s_t = modify_state(s_t)
        r_s_t_next = modify_state(s_t_next)
        self.last_inputs = np.expand_dims(np.array(r_s_t).flatten(), axis=0)
        self.data_inputs = np.expand_dims(np.array(r_s_t_next).flatten(), axis=0)
        self.d_t = d_t
        self.r_t = (np.array([0.6, 0.3, 0.1]) @ r_t)
        self.a_t = a_t

        population_vectors = pygad.gann.population_as_vectors(
            population_networks=self.GANN_instance.population_networks)
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
    """
    Implementation of a DDQN agent to determine the next target of for the UAV

    INPUTS:
        nS -> number of state space variables when state is flattened (int)
        nA -> number of action space output (int)
        epsilon_i & epsilon_f -> boundaries for epsilon decay of function (max float, min float)
                --> See decay_epsilon() function for further explanation
        alpha -> model learning rate assign at initialization (float)
        gamma -> contribution of target model for target updates (float)
        epsilon -> initialized value for epsilon (float)
    """

    def __init__(self, nS: int, nA: int, epsilon_i: float = 1.0, epsilon_f: float = 0.0,
                 alpha: float = 0.001, gamma: float = 0.95, epsilon: float = 0.5, mem_len: int = 2500):
        # ADF 2.0
        self.nS = nS
        self.nA = nA
        # ADF 1.0

        self.memory = deque([], mem_len)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f

        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights

        self.loss = []

    def build_model(self):
        """
        Build the sequential layers of both DDQN models.
        Called at initialization with no input.

        OUTPUT:
            model -> returns an agent for the ddqn class
        """
        model = tf.keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(tf.keras.layers.Input(shape=(self.nS,)))
        model.add(tf.keras.layers.Dense(256, activation='relu'))  # [Input] -> Layer 1
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(96, activation='relu'))  # Layer 1 -> 2
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.nA, activation='softmax'))  # Layer 2 -> [output]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
                      loss='mean_squared_error',  # Loss function: Mean Squared Error
                      metrics=['accuracy'])  # Optimaizer: Adam (Feel free to check other options)
        return model

    def update_learning_rate(self, new_learning_rate):
        """
        Updates the learning rate of the DDQN model by adjusting the optimizer learning rate.
        Enables the use of warmup logic when training model

        INPUT:
            new_learning_rate -> value to change to which the learning rate is adjusted

        OUTPUT:
            change in model learning rate (no return)
        """
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_learning_rate)

    def decay_epsilon(self, n: float):
        """
        Directly sets exploration rate according to n with 0 meaning 'just started' and 1 meaning 'done'.
        Using maximum of epsilon_f and epsilon_i, these values are max and min exploitation probability.

        INPUT:
            n -> value on scale [0,1] that mutates the upper limit to determine exploration change (float)

        OUTPUT:
            change in exploration rate (no return)
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - n * (self.epsilon_i - self.epsilon_f))

    def update_target_from_model(self):
        """
        Copy the weights from the primary model to the target model.
        Use at intervals that create learning "checkpoints" for the model.
        """
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def act(self, state):
        """
        Equivalent to "forward". Uses the provided state to determine the action.
        Action is gated by explore/exploit change.
        If random float is less than epsilon, explore: Else, exploit.

        INPUT:
            state -> environment state (potentially unknown values)

        OUTPUT:
            action -> value projected on actin space (int)
        """
        r_state = modify_state(state)
        if np.random.rand() < self.epsilon:
            # Explore: Make random prediction on action space
            return np.random.randint(self.nA)

        # Exploit: Use the NN to predict the correct action from this state
        action_vals = self.model.predict(np.expand_dims(np.array(r_state).flatten(), axis=0))
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        """
        Equivalent to "forward" and "act". Uses the provided state to determine the action.
        Action is not gated by explore/exploit change leading to model decisions.

        INPUT:
            state -> environment state (potentially unknown values)

        OUTPUT:
            action -> value projected on actin space (int)
        """
        r_state = modify_state(state)
        # Exploit: Use the NN to predict the correct action from this state
        action_vals = self.model.predict(np.expand_dims(np.array(r_state).flatten(), axis=0))
        return np.argmax(action_vals[0])

    def update_mem(self, state, action, reward, nstate, done, step):
        """
        Stores current performance period on the memory stack

        """
        # Store the experience in memory
        r_state = modify_state(state)
        r_nstate = modify_state(nstate)
        self.memory.append((r_state, action, reward, r_nstate, done, step))

    def train(self, batch_size):
        # We can use the change in weights from the target network to current network for stability
        weights = []
        for layer in self.model.layers:
            for weight in layer.weights:
                weights.append(weight.numpy().flatten())
        np_weights = np.concatenate(weights)

        weights_target = []
        for layer in self.model_target.layers:
            for weight in layer.weights:
                weights_target.append(weight.numpy().flatten())
        np_weights_target = np.concatenate(weights_target)

        # Output is number from [0,1]
        avg_weight_diff = np.average((np_weights - np_weights_target) / (np_weights + np_weights_target))
        avg_sqr_diff = avg_weight_diff / abs(avg_weight_diff) * avg_weight_diff ** 2

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

        for state, action, reward, nstate, done, step in minibatch:
            x.append(np.expand_dims(np.array(state).flatten(), axis=0))
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]

            if np.array(reward).mean() <= 0.0:
                target = self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)] - avg_sqr_diff
            elif done:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = (np.array([0.7, 0.3, 0]) @ reward) - avg_sqr_diff
            else:  # Non terminal, Using Q to get T is Double DQN
                target = ((np.array([0.7, 0.3, 0]) @ reward) +
                          self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)]
                          - avg_sqr_diff)

            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1

        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)

        loss = self.model.train_on_batch(x_reshape, y_reshape)
        return loss



class get_ddqn_agentp():
    def __init__(self, env, nS: int, nA: int, epsilon_i: float = 1.0, epsilon_f: float = 0.0,
                 alpha: float = 0.001, gamma: float = 0.95, epsilon: float = 0.5, mem_len: int = 2500):
        # ADF 2.0
        self.nS = nS
        self.nA = nA

        self.state1_max = env.dim
        self.state2_max = env.max_num_steps

        self.memory = deque([], mem_len)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f

        self.model = self.build_model()
        self.model_target = self.build_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights
        self.loss = []

    def build_model(self):
        model = tf.keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(tf.keras.layers.Input(shape=(self.nS,)))
        model.add(tf.keras.layers.Dense(32, activation='relu'))  # [Input] -> Layer 1
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(16, activation='relu'))  # Layer 2 -> 3
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.nA, activation='softmax'))  # Layer 3 -> [output]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha),
                      loss='mean_squared_error',  # Loss function: Mean Squared Error
                      metrics=['accuracy'])  # Optimaizer: Adam (Feel free to check other options)
        return model

    def decay_epsilon(self, n):
        """
        Decays the get_ddqn_agent's exploration rate according to n, which is a
        float between 0 and 1 describing how far along training is,
        with 0 meaning 'just started' and 1 meaning 'done'.
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - n * (self.epsilon_i - self.epsilon_f))

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def act(self, state):
        r_state = [state[0] / self.state1_max, state[1] / 6_800_000,
                   state[2] / self.state2_max, state[3] / self.state2_max]
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nA)

        # Exploit: Use the NN to predict the correct action from this state
        action_vals = self.model.predict(np.reshape(np.array(r_state), (-1, self.nS)))
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        r_state = [state[0] / self.state1_max, state[1] / 6_800_000,
                   state[2] / self.state2_max, state[3] / self.state2_max]
        action_vals = self.model.predict(np.reshape(np.array(r_state), (
        -1, self.nS)))  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def update_mem(self, state, action, reward, nstate, done):
        # Store the experience in memory
        r_state = [state[0] / self.state1_max, state[1] / 6_800_000,
                   state[2] / self.state2_max, state[3] / self.state2_max]
        r_nstate = [nstate[0] / self.state1_max, nstate[1] / 6_800_000,
                    nstate[2] / self.state2_max, nstate[3] / self.state2_max]
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
            x.append(np.expand_dims(np.array(state).flatten(), axis=0))
            # Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = (np.array([0.25, 0.25, 0.5])) @ reward
            else:  # Non terminal, Using Q to get T is Double DQN
                target = (np.array([0.25, 0.25, 0.5]) @ reward +
                          self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)])

            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1

        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size, self.nS)
        y_reshape = np.array(y)

        loss = self.model.train_on_batch(x_reshape, y_reshape)
        return loss
