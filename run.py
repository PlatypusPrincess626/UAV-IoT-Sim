import math
import os
import argparse
from time import time, sleep
from typing import Optional, Tuple
import csv
import datetime
import numpy as np
import os
import tensorflow as tf
import wandb

# custom dependencies
from UAV_IoT_Sim import UAV_IoT_Sim
from env_utils import model_utils
from env_utils.logger_utils import RunningAverage, get_logger


# os.environ["LD_LIBRARY_PATH"] = "/UAV-IoT-Sim/uav-iot-env/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
# env LD_LIBRARY_PATH=/UAV-IoT-Sim/uav-iot-env/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9 python run.py --project-name DQNN042024 --steps 50_000 --eval-frequency 1_000


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
    info = None
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
            train_model, old_state, old_action, comms, move, harvest = eval_env.step(agent)
            buffer_done = eval_env.terminated
            info = eval_env.curr_info
            crashed = eval_env.terminated

            if buffer_done or eval_env.truncated:
                done = True

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

            if train_model or done:
                # if True:
                #     agent.update(old_state, old_action, eval_env.curr_reward, eval_env.curr_state, buffer_done)
                # DDQN
                agent.update_mem(old_state, old_action, eval_env.full_reward, eval_env.curr_state, buffer_done)

            ep_reward += info.get("Reward_Change")

            if log_metrics and i == eval_episodes - 1:
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

        if log_metrics and i == eval_episodes - 1:
            filename = ("sens_pts0_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + ".csv")
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(SensCoords)

        if log_metrics and i == eval_episodes - 1:
            filename = ("cluster_pts0_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + ".csv")
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(CHCoords)

        if log_metrics and i == eval_episodes - 1:
            print(eval_env.ch_sensors)
            filename = ("age_metrics0_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + ".csv")
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(CH_Age)

        if log_metrics and i == eval_episodes - 1:
            filename = ("data_metrics0_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + ".csv")
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(CH_Data)

        if log_metrics and i == eval_episodes - 1:
            filename = ("uav_metrics0_" + curr_date_time.strftime("%d") + "_" +
                        curr_date_time.strftime("%m") + ".csv")
            open(filename, 'x')
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter='|')
                csvwriter.writerows(UAV_Metrics)

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
        total_reward += ep_reward / (eval_env.curr_step + count)
        total_steps += eval_env.curr_step

    return 1 - (num_crashes / eval_episodes), total_reward / eval_episodes, \
           total_steps / eval_episodes, accum_avgAoI / eval_episodes, accum_peakAoI / eval_episodes, \
           accum_dataDist / eval_episodes, 1000 * accum_dataColl / eval_episodes, CH_Metrics, \
           accum_comms / eval_episodes, accum_move / eval_episodes, accum_harvest / eval_episodes


def train(
        agent,
        env: object,
        env_str: str,
        total_steps: int,
        eval_frequency: int,
        eval_episodes: int,
        policy_path: str,
        logger=None,
):
    start_time = time()
    # agent.eval_off()
    env.reset()
    sr, ret, length = 0.0, 0.0, 0.0
    for timestep in range(total_steps):
        done = step(agent, env)
        if done:
            if len(agent.memory) > 128:
                agent.train(128)
        # QL
        agent.decay_epsilon(timestep / total_steps)

        if timestep % (2 * eval_frequency) == 0:
            # DDQN
            agent.update_target_from_model()
            env.reset()

        if timestep % eval_frequency == 0:
            sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
                comms, move, harvest = evaluate(agent, env, eval_episodes, False)

        print(
            f"Training Steps: {timestep}, Env: {env_str}, Sucess Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}"
        )

    sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
        comms, move, harvest = evaluate(agent, env, eval_episodes, True, env_str, start_time)



def step(agent, env):
    train_model, old_state, old_action, comms, move, harvest = env.step(agent)
    buffer_done = env.terminated
    done = False

    if buffer_done or env.truncated:
        done = True

    if train_model or done:
        # if True:
        # QL
        # agent.update(old_state, old_action, env.curr_reward, env.curr_state, buffer_done)
        # DDQN
        agent.update_mem(old_state, old_action, env.full_reward, env.curr_state, buffer_done)
    return done


def prepopulate(agent, prepop_steps, env):
    # QL
    agent.decay_epsilon(0)

    timestep = 0
    while timestep < prepop_steps:
        env.reset()
        done = False

        while not done:
            print(f"Prepop Step: {timestep}")
            train_model, old_state, old_action, comms, move, harvest = env.step(agent)
            buffer_done = env.terminated

            if buffer_done or env.truncated:
                done = True

            if train_model or done:
                agent.update_mem(old_state, old_action, env.full_reward, env.curr_state, buffer_done)
            timestep += 1

        if len(agent.memory) > 128:
            agent.train(128)


def run_experiment(args):
    env_str = args.env
    print("Creating Evironment")
    env = UAV_IoT_Sim.make_env(scene=env_str, num_sensors=50, num_ch=5, num_uav=1, max_num_steps=720)

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # V1
    # agent = model_utils.get_ddqn_agent(
    #     env
    # )
    # DDQN

    ### V2: for under complex systems
    # Train agent to determine charging or travelling
    agent = model_utils.get_ddqn_agent(
        env,
        ((env.num_ch + 3) * 3),
        env.num_ch + 5
    )

    policy_save_dir = os.path.join(
        os.getcwd(), "policies", args.project_name
    )
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}"
    )

    prepopulate(agent, 72000, env)

    print("Beginning Training")
    train(
        agent,
        env,
        args.env,
        args.steps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path
    )


if __name__ == "__main__":
    run_experiment(get_args())
