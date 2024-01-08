import os
import argparse
from collections import deque

# custom dependencies
from UAV_IoT_Sim import UAV_IoT_Sim
from env_utils import model_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name",
        type=str,
        default="uav-iot-sim-test",
        help="The project name (for loggers) to store results."        
    )
    parser.add_argument(
        "--env",
        type=str,
        default="test",
        help="The scene for the project to take place."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Q-Learning",
        help="ML model to use for testing."
    )
    parser.add_argument(
        "--life",
        type=int,
        default=720,
        help="Maximum number of steps in the episode."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default = 100_000,
        help="Maximum number of steps for training."
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10_000,
        help="Frequency of agent evaluation."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default = 3,
        help="Episodes to evaluate each evaluation period."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10_000,
        help="Dimensions of square (dim x dim) area."
    )
    parser.add_argument(
        "--uavs",
        type=int,
        default=1,
        help="The number of UAVs on site."
    )
    parser.add_argument(
        "--clusterheads",
        type=int,
        default=5,
        help="The number of clusterheads."
    )
    parser.add_argument(
        "--sensors",
        type=int,
        default=50,
        help="The number of sensors spread across the environment."
    )
    return parser.parse_args()

class RunningAverage:
    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.sum = 0

    def add(self, val):
        self.q.append(val)
        self.sum += val
        if len(self.q) > self.size:
            self.sum -= self.q.popleft()

    def mean(self):
        # Avoid divide by 0
        return self.sum / max(len(self.q), 1)

def evaluate(
    agent,
    eval_env,
    eval_episodes,
):
    
    total_reward = 0
    num_crashes = 0
    total_steps = 0
    
    for _ in range(eval_episodes):
        eval_env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            curr_obs = eval_env._curr_state
            obs_next, reward, terminated, truncated, info = eval_env.step(agent, test=True)
            action = info.get("Last_Action", None)
            agent.update(obs_next, action, reward, curr_obs, done)
            ep_reward =+ reward
            
        total_reward += ep_reward
        total_steps += eval_env._curr_step
        if info.get("Crashed", False):
            num_crashes += 1

    return num_crashes/total_steps, total_reward, total_steps

def train(
    agent,
    env: object,
    env_str: str,
    total_steps: int,
    eval_frequency: int,
    eval_episodes: int,
    policy_path: str,
    mean_success_rate: RunningAverage,
    mean_episode_length: RunningAverage,
    mean_reward: RunningAverage
):
    #agent.eval_off()
    env.reset()
    
    for timestep in range(total_steps):
        done = step(agent, env)
    
        if done:
            env.reset()
        
        if timestep % eval_frequency == 0:
            sr, ret, length = evaluate(agent, env, eval_episodes)
            
        
        print(
            f"Training Steps: {timestep}, Env: {env_str}, Crash Rate: {sr:.2f}, \
            Return: {ret:.2f}, Episode Length: {length:.2f}"
        )
        
def step(agent, env):
    obs_curr = env._curr_state
    obs_next, reward, terminated, truncated, info = env.step(agent)
    action = info.get("Last_Action", None)
    
    if truncated:
        buffer_done = False
    else:
        buffer_done = terminated
      
    agent.update(obs_next, action, reward, obs_curr, buffer_done)
    return terminated
        
def run_experiment(args):
    env_str = args.env
    env = UAV_IoT_Sim.make_env(env_str)
    #device = torch.device("cuda")
    
    agent = model_utils.get_ql_agent(
        env
    )

    policy_save_dir = os.path.join(
        os.getcwd(), "policies", args.project_name
    )
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}"
    )
    
    if os.path.exists(policy_path + "_mini_checkpoint.pt"):
        steps_completed = agent.load_mini_checkpoint(policy_path)["step"]
        print(
            f"Found a mini checkpoint that completed {steps_completed} training steps."
        )
    else:
        # Prepopulate the replay buffer
        # prepopulate(agent, 50_000, envs)
        mean_success_rate = RunningAverage(10)
        mean_reward = RunningAverage(10)
        mean_episode_length = RunningAverage(10)
        
    train(
        agent,
        env,
        args.env,
        args.steps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        mean_success_rate,
        mean_episode_length,
        mean_reward
    )

if __name__ == "__main__":
    run_experiment(get_args())