import numpy as np
import tensorflow as tf
import pygad
import pygad.nn
import pygad.gann
import pandas as pd
import statistics as stats

from env_utils.logger_utils import RunningAverage


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

        self.num_actions = env._num_ch
        self.num_states = (6800) + (env._num_uav + env._num_ch) * (25 * env._max_steps) * (env._max_steps)
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
                if caparison.all():
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
        self._num_inputs = 3*(env._num_ch+1)
        self._num_output = env._num_ch
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
        self.last_inputs = np.expand_dims(np.array(s_t).flatten(), axis=0)
        self.data_inputs = np.expand_dims(np.array(s_t_next).flatten(), axis=0)
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
        inputs = np.expand_dims(np.array(s_t).flatten(), axis=0)

        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[self.sol_idx],
                                       data_inputs=inputs)

        return np.argmax(predictions)


class DDDQN(tf.keras.Model):
    def __init__(self):
        super(DDDQN, self).__init__(env)
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(env._num_ch, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a


class exp_replay():
    def __init__(self, env, buffer_size=25000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *((env._num_ch+1)*3)), dtype=np.float32)
        self.action_mem = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *((env._num_ch+1)*3)), dtype=np.float32)
        self.done_mem = np.zeros(self.buffer_size, dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size=64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


class get_ddqn_agent:
    def __init__(self, env, gamma=0.99, replace=100, lr=0.001):
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        self.replace = replace
        self.trainstep = 0
        self.memory = exp_replay(env)
        self.batch_size = 64
        self.q_net = DDDQN(env)
        self.target_net = DDDQN(env)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)

        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(env._num_ch)])

        else:
            actions = self.q_net.advantage(np.array(state).flatten())
            action = np.argmax(actions)
            return action

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(np.array(state).flatten(), action, reward, np.array(next_state).flatten(), done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def epsilon_decay(self, new_epsilon):
        self.epsilon = new_epsilon
        return self.epsilon

    def train(self):
        if self.memory.pointer < self.batch_size:
            return

        if self.trainstep % self.replace == 0:
            self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)  # optional
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
        self.q_net.train_on_batch(states, q_target)
        self.trainstep += 1

        self.qvalue_max.add(np.argmax(target))
        self.qvalue_mean.add(np.mean(target))
        self.qvalue_min.add(np.argmin(target))

        self.target_max.add(np.argmax(next_state_val))
        self.target_mean.add(np.mean(next_state_val))
        self.target_min.add(np.argmin(next_state_val))

        loss = (np.square(np.argmax(next_state_val) - np.argmax(target)))
        self.td_errors.add(loss)