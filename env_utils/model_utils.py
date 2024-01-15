import numpy as np
import pygad
import pygad.nn
import pygad.gann
import pandas as pd

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
        Decays the agent's exploration rate according to n, which is a
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
        Uses the q-learning update rule to update the agent's predictions
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

        self.a_t = 0
        self.r_t = 0
        self.d_t = False
        self.gamma = 0.95
        self.last_inputs = np.array([0]*self._num_inputs)
        self.data_inputs = np.array([0]*self._num_inputs)

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

        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_network[sol_idx],
                                       data_inputs=self.data_inputs)
        previous = pygad.nn.predict(last_layer=self.GANN_instance.population_network[sol_idx],
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
        self.last_inputs = np.flatten(np.array(s_t))
        self.data_inputs = np.flatten(np.array[s_t_next])
        self.d_t = d_t
        self.r_t = r_t
        self.a_t = a_t

        population_vectors = pygad.gann.population_as_vectors(population_networks=self.GANN_instance.population_networks)
        initial_population = population_vectors.copy()

        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_network[self.sol_idx],
                                       data_inputs=self.data_inputs)
        previous = pygad.nn.predict(last_layer=self.GANN_instance.population_network[self.sol_idx],
                                    data_inputs=self.last_inputs)

        self.qvalue_max.add(previous.max().item())
        self.qvalue_mean.add(previous.mean().item())
        self.qvalue_min.add(previous.min().item())

        self.target_max.add(predictions.max().item())
        self.target_mean.add(predictions.mean().item())
        self.target_min.add(predictions.min().item())

        loss = (np.square(previous - predictions)).mean()
        self.td_errors.add(loss.item())

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

        self.sol, self.sol_fit, self.sol_idx = ga_instance.best_solution()

    def act(self, s_t):
        inputs = np.flatten(np.array(s_t))

        predictions = pygad.nn.predict(last_layer=self.GANN_instance.population_networks[self.sol_idx],
                                       data_inputs=inputs)

        return np.argmax(predictions)
