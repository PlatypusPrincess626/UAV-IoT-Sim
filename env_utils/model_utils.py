import numpy as np
from logger_utils import RunningAverage


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

    def decay_epsilon(self, n):
        """
        Decays the agent's exploration rate according to n, which is a
        float between 0 and 1 describing how far along training is, 
        with 0 meaning 'just started' and 1 meaning 'done'.
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))

    def act(self, s_t):
        """
        Epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        return np.argmax(self.Q[s_t])

    def update(self, s_t, a_t, r_t, s_t_next, d_t):
        """
        Uses the q-learning update rule to update the agent's predictions
        for Q(s_t, a_t).
        """
        Q_next = np.max(self.Q[s_t_next])

        targets = r_t + (1 - d_t) * self.gamma * Q_next

        self.qvalue_max.add(self.Q[s_t].max().item())
        self.qvalue_mean.add(self.Q[s_t].mean().item())
        self.qvalue_min.add(self.Q[s_t].min().item())

        self.target_max.add(targets.max().item())
        self.target_mean.add(targets.mean().item())
        self.target_min.add(targets.min().item())

        loss = (np.square(self.Q[s_t]-targets)).mean(axis=1)
        self.td_errors.add(loss.item())

        self.Q[s_t, a_t] = self.Q[s_t, a_t] + self.alpha * \
                           (r_t + (1 - d_t) * self.gamma * Q_next - self.Q[s_t, a_t])
