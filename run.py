import os
import argparse
from time import time, sleep
from typing import Optional, Tuple

import wandb

# custom dependencies
from UAV_IoT_Sim import UAV_IoT_Sim
from env_utils import model_utils
from env_utils.logger_utils import RunningAverage, get_logger


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
        default=100_000,
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
        default=3,
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
    parser.add_argument(
        "--disable_wandb",
        type=bool,
        default=False,
        help="Activate wandb."
    )
    return parser.parse_args()


def evaluate(
        agent,
        eval_env,
        eval_episodes,
):
    total_reward = 0
    num_crashes = 0
    total_steps = 0
    #agent.decay_epsilon(1)

    for _ in range(eval_episodes):
        eval_env.reset()
        done = False
        ep_reward = 0
        avgAoI = 0.0
        peakAoI = 0.0
        dataDist = 0.0
        dataColl = 0.0

        while not done:
            curr_obs = eval_env._curr_state
            obs_next, reward, terminated, truncated, info = eval_env.step(agent)

            if terminated or truncated:
                done = True
            action = info.get("Last_Action", None)
            avgAoI = info.get("Avg_Age", 0.0)
            peakAoI = info.get("Peak_Age", 0.0)
            dataDist = info.get("Data_Distribution", 0.0)
            dataColl = info.get("Total_Data_Change", 0.0)
            agent.update_mem(curr_obs, action, reward, obs_next, done)
            if len(agent.memory)>64:
                agent.train(64)
            ep_reward = + reward

        agent.update_target_from_model()
        total_reward += ep_reward
        total_steps += eval_env._curr_step
        if info.get("Crashed", False):
            num_crashes += 1

    if total_steps == 0:
        total_steps = 1
    return (1-num_crashes/total_steps), total_reward, total_steps, avgAoI, peakAoI, dataDist, dataColl


def train(
        agent,
        env: object,
        env_str: str,
        total_steps: int,
        eval_frequency: int,
        eval_episodes: int,
        policy_path: str,
        logger,
        mean_success_rate: RunningAverage,
        mean_episode_length: RunningAverage,
        mean_reward: RunningAverage
):
    start_time = time()
    # agent.eval_off()
    env.reset()
    sr, ret, length = 0.0, 0.0, 0.0
    for timestep in range(total_steps):
        print(f"Step {timestep}: ")
        done = step(agent, env)
        #agent.decay_epsilon(timestep / total_steps)

        if done:
            agent.update_target_from_model()
            env.reset()

        if timestep % eval_frequency == 0:
            hours = (time() - start_time) / 3600
            log_vals = {
                "losses/TD_Error": agent.td_errors.mean(),
                "losses/Grad_Norm": agent.grad_norms.mean(),
                "losses/Max_Q_Value": agent.qvalue_max.mean(),
                "losses/Mean_Q_Value": agent.qvalue_mean.mean(),
                "losses/Min_Q_Value": agent.qvalue_min.mean(),
                "losses/Max_Target_Value": agent.target_max.mean(),
                "losses/Mean_Target_Value": agent.target_mean.mean(),
                "losses/Min_Target_Value": agent.target_min.mean(),
                "losses/hours": hours,
            }
            sr, ret, length, avgAoI, peakAoI, dataDist, dataColl = evaluate(agent, env, eval_episodes)

            log_vals.update(
                {
                    f"{env_str}/SuccessRate": sr,
                    f"{env_str}/Return": ret,
                    f"{env_str}/EpisodeLength": length,
                    f"{env_str}/AverageAoI": avgAoI,
                    f"{env_str}/PeakAoI": peakAoI,
                    f"{env_str}/Distribution": dataDist,
                    f"{env_str}/TotalCollected": dataColl,
                }
            )

            logger.log(
                log_vals,
                step=timestep,
            )

        print(
            f"Training Steps: {timestep}, Env: {env_str}, Sucess Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}"
        )


def step(agent, env):
    obs_curr = env._curr_state
    obs_next, reward, terminated, truncated, info = env.step(agent)
    action = info.get("Last_Action", None)

    buffer_done = terminated
    done = False
    if terminated or truncated:
        done = True

    agent.update_mem(obs_next, action, reward, obs_curr, buffer_done)
    if len(agent.memory) > 64:
        agent.train(64)
    return done

def prepopulate(agent, prepop_steps, env):
    timestep = 0
    while timestep < prepop_steps:
        env.reset()
        print(f"Prepop Step: {timestep}")
        done = False
        while not done:
            #agent.decay_epsilon(0)
            obs_curr = env._curr_state
            obs_next, reward, terminated, truncated, info = env.step(agent)
            action = info.get("Last_Action", None)

            buffer_done = terminated
            if terminated or truncated:
                agent.update_target_from_model()
                done = True
            agent.update_mem(obs_next, action, reward, obs_curr, buffer_done)
            if len(agent.memory) > 64:
                agent.train(64)
            timestep += 1

def run_experiment(args):
    env_str = args.env
    print("Creating Evironment")
    env = UAV_IoT_Sim.make_env(env_str)
    # device = torch.device("cuda")

    print("Creating Agent")
    agent = model_utils.get_ddqn_agent(
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

    wandb_kwargs = {"resume": None}
    logger = get_logger(policy_path, args, wandb_kwargs)

    prepopulate(agent, 50_000, env)
    mean_success_rate = RunningAverage(10)
    mean_reward = RunningAverage(10)
    mean_episode_length = RunningAverage(10)

    print("Beginning Training")
    train(
        agent,
        env,
        args.env,
        args.steps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        logger,
        mean_success_rate,
        mean_episode_length,
        mean_reward
    )


if __name__ == "__main__":
    run_experiment(get_args())
