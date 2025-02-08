import math
import os
import argparse
from time import time, sleep
from typing import Optional, Tuple
import csv
import datetime

import wandb

# custom dependencies
from UAV_IoT_Sim import UAV_IoT_Sim
from env_utils import model_utils
from env_utils.logger_utils import RunningAverage, get_logger

import os
import tensorflow as tf
#os.environ["LD_LIBRARY_PATH"] = "/UAV-IoT-Sim/uav-iot-env/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
#env LD_LIBRARY_PATH=/UAV-IoT-Sim/uav-iot-env/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9 python run.py --project-name DQNN042024 --steps 50_000 --eval-frequency 1_000


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
        default=True,
        help="Activate wandb."
    )
    return parser.parse_args()


def evaluate(
        agent,
        eval_env,
        eval_episodes,
        log_metrics=False,
        env_str=None,
        logger=None,
        start_time=None
):
    total_reward = 0
    num_crashes = 0
    total_steps = 0
    count = 0

    accum_avgAoI = 0
    accum_peakAoI = 0
    accum_dataDist = 0
    accum_dataColl = 0

    CH_Metrics = [[0, 0] for _ in range(eval_env.num_ch)]

    CH_Age = []
    CH_Data = []
    UAV_Metrics = []
    CHCoords = []
    SensCoords = []

    accum_comms = 0
    accum_move = 0
    accum_harvest = 0

    # QL
    # for agent in agents:
    agent.decay_epsilon(1)

    for i in range(eval_episodes):
        eval_env.reset()

        done = False
        crashed = False
        ep_reward = 0
        avgAoI = 0.0
        peakAoI = 0.0
        dataDist = 0.0
        dataColl = 0.0
        CH_Metrics = [[0, 0] for _ in range(eval_env.num_ch)]

        while not done:
            train_model, train_CH, old_state, old_action, comms, move, harvest = eval_env.step(agent)
            buffer_done = eval_env.terminated
            info = eval_env.curr_info
            crashed = eval_env.terminated

            if buffer_done or eval_env.truncated:
                done = True

            print(eval_env.full_reward)

            avgAoI += info.get("Avg_Age", 0.0)
            peakAoI += info.get("Peak_Age", 0.0)
            dataDist += info.get("Data_Distribution", 0.0)
            dataColl += info.get("Total_Data_Change", 0.0)
            accum_comms += comms
            accum_move += move
            accum_harvest += harvest

            ch: int
            for ch in range(len(CH_Metrics)):
                CH_Metrics[ch][0] = eval_env.curr_state[ch + 1][1]
                CH_Metrics[ch][1] = eval_env.curr_state[ch + 1][2]

            if (train_model or eval_env.truncated) and not buffer_done:
                # agent.update(old_state, old_action, eval_env.archived_rewards,
                #              eval_env.curr_state, buffer_done, eval_env.curr_step)
                # DDQN
                agent.update_mem(old_state, old_action, eval_env.archived_rewards,
                                 eval_env.curr_state, buffer_done, eval_env.curr_step)

            ep_reward += eval_env.full_reward

            if log_metrics and i == eval_episodes-1:
                CH_Age.append([CH_Metrics[0][1], CH_Metrics[1][1], CH_Metrics[2][1],
                               CH_Metrics[3][1], CH_Metrics[4][1]])
                CH_Data.append([eval_env.curr_state[0][1], CH_Metrics[0][0], CH_Metrics[1][0], CH_Metrics[2][0],
                               CH_Metrics[3][0], CH_Metrics[4][0]])
                UAV_Metrics.append([eval_env.uavX, eval_env.uavY, eval_env.curr_state[0][2], comms, move, harvest])


        curr_date_time = datetime.datetime.now()

        if log_metrics and i == eval_episodes - 1:
            for sensor in range(len(eval_env.sensX)):
                SensCoords.append([eval_env.sensX[sensor], eval_env.sensY[sensor]])
            for cluster in range(len(eval_env.chX)):
                CHCoords.append([eval_env.chX[cluster], eval_env.chY[cluster]])

        csv_str = "_Orig5K.csv"

        if log_metrics and i == eval_episodes - 1:
            filename = ("sens_pts_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + csv_str)
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(SensCoords)

        if log_metrics and i == eval_episodes - 1:
            filename = ("cluster_pts_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + csv_str)
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(CHCoords)

        if log_metrics and i == eval_episodes - 1:
            print(eval_env.ch_sensors)
            filename = ("age_metrics_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + csv_str)
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(CH_Age)

        if log_metrics and i == eval_episodes - 1:
            filename = ("data_metrics_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + csv_str)
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(CH_Data)

        if log_metrics and i == eval_episodes - 1:
            filename = ("uav_metrics_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + csv_str)
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(UAV_Metrics)

        # DDQN
        # for agent in agents:
        if len(agent.memory) > 2048:
            agent.train(2048)

        accum_avgAoI += avgAoI / (eval_env.curr_step + count)
        accum_peakAoI += peakAoI / (eval_env.curr_step + count)
        accum_dataDist += dataDist / (eval_env.curr_step + count)
        accum_dataColl += dataColl / (eval_env.curr_step + count)

        for ch in range(len(CH_Metrics)):
            CH_Metrics[ch][0] /= eval_env.curr_step
            CH_Metrics[ch][1] /= eval_env.curr_step

        accum_comms /= eval_env.curr_step
        accum_move /= eval_env.curr_step
        accum_harvest /= eval_env.curr_step

        if crashed:
            num_crashes += 1
        total_reward += ep_reward / (eval_env.curr_step+count)
        total_steps += eval_env.curr_step


    
    return 1 - (num_crashes / eval_episodes), total_reward / eval_episodes, \
        total_steps / eval_episodes, accum_avgAoI / eval_episodes, accum_peakAoI / eval_episodes, \
        accum_dataDist / eval_episodes, 1000 * accum_dataColl / eval_episodes, CH_Metrics, \
        accum_comms/eval_episodes, accum_move/eval_episodes, accum_harvest/eval_episodes


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
        mean_reward: RunningAverage,
        logger=None,
):
    start_time = time()
    env.reset()
    sr, ret, length = 0.0, 0.0, 0.0

    agent.decay_epsilon(1)

    for timestep in range(total_steps):
        done = step(agent, env)

        if done:
        #     # for agent in agents:
            if len(agent.memory) > 2048:
                agent.train(2048)

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
            sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
                comms, move, harvest = evaluate(agent, env, eval_episodes)

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

            agent.update_target_from_model()
            env.reset()


        print(
            f"Training Steps: {timestep}, Env: {env_str}, Sucess Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}"
        )

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
    sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
        comms, move, harvest = evaluate(agent, env, eval_episodes, True, env_str, start_time)

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


def step(agent, env):
    train_model, train_CH, old_state, old_action, comms, move, harvest = env.step(agent)
    buffer_done = env.terminated
    done = False

    if buffer_done or env.truncated:
        done = True

    if (train_model or env.truncated) and not buffer_done:
        print(f"Training")
        #QL/GANN
        # agent.update(old_state, old_action, env.archived_rewards,
        #              env.curr_state, buffer_done, env.curr_step)
        # DDQN
        agent.update_mem(old_state, old_action, env.archived_rewards,
                         env.curr_state, buffer_done, env.curr_step)
    return done


def prepopulate(agent, prepop_steps, env, eval_frequency):
    timestep = 0
    # QL
    agent.decay_epsilon(0)

    while timestep < prepop_steps:
        env.reset()
        done = False

        while not done:
            print(f"Prepop Step: {timestep}, Reward: {env.full_reward}")
            train_model, train_CH, old_state, old_action, comms, move, harvest = env.step(agent)
            buffer_done = env.terminated

            if buffer_done or env.truncated:
                done = True

            if (train_model or env.truncated) and not buffer_done:
                agent.update_mem(old_state, old_action, env.archived_rewards,
                                 env.curr_state, buffer_done, env.curr_step)

                # QL/GANN Agents
                # agent.update(old_state, old_action, env.archived_rewards,
                #                  env.curr_state, buffer_done, env.curr_step)
            timestep += 1

        if len(agent.memory) > 2048:
            agent.train(2048)

        if timestep % eval_frequency == 0:
            # DDQN
            # for agent in agents:
            agent.update_target_from_model()
            env.reset()

def run_experiment(args):
    env_str = args.env
    print("Creating Evironment")
    env = UAV_IoT_Sim.make_env(scene=env_str, num_sensors=50, num_ch=5, num_uav=1, max_num_steps=720)
    
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Other
    # agent = model_utils.get_ddqn_agent(
    #     env
    # )
    # DDQN
    # agents = []
    # for i in range(env.num_ch):
    agent = model_utils.get_ddqn_agent(
        env,
        ((env.num_ch + 2) * 3),
        env.num_ch
    )
    # agents.append(agent)

    policy_save_dir = os.path.join(
        os.getcwd(), "policies", args.project_name
    )
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}"
    )

    prepopulate(agent, 50_000, env, args.eval_frequency)
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
        mean_success_rate,
        mean_episode_length,
        mean_reward,
        # logger
    )


if __name__ == "__main__":
    run_experiment(get_args())
