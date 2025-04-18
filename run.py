import math
import os
import argparse
from asyncio import new_event_loop
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
        agent_p,  # Used for Dual Model/ Pass None for single model
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
    """Dual Agent Systems"""
    agent_p.decay_epsilon(1)
    """END"""
    switch = True

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
            """Single DRL"""
            # train_model, train_CH, old_state, old_action, comms, move, harvest = eval_env.step(agent)
            # print(eval_env.full_reward)
            """Multiple DRL"""
            train_model, train_p, old_state, old_action, action_p, old_pstate, comms, move, harvest = (
                eval_env.step(agent, agent_p))
            print(eval_env.full_reward)
            """END"""

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

            if (train_model or eval_env.truncated) and not buffer_done:
                # agent.update(old_state, old_action, eval_env.archived_rewards,
                #              eval_env.curr_state, buffer_done, eval_env.curr_step)
                # DDQN
                agent.update_mem(old_state, old_action, eval_env.archived_rewards,
                                 eval_env.curr_state, buffer_done, eval_env.curr_step)

            """Dual Model Training Addition"""
            if train_p or done:
                agent_p.update_mem(old_pstate, int(action_p), eval_env.archived_rewardsp,
                                   eval_env.curr_pstate, buffer_done)
            """END"""

            ep_reward += eval_env.full_reward

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

        csv_str = ("_Dual_NForced_100K_3K_2.csv")

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
        if switch:
            if len(agent.memory) > 2500:
                agent.train(2500)
            switch = not switch
        # """Dual Agent Systems"""
        else:
            if len(agent_p.memory) > 2500:
                agent_p.train(2500)
            switch = not switch
        """END"""

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
        agent_p,  # Pass None if using Single Agent
        env: object,
        env_str: str,
        total_steps: int,
        eval_frequency: int,
        eval_episodes: int,
        policy_path: str,
        logger=None,
):
    start_time = time()
    env.reset()
    sr, ret, length = 0.0, 0.0, 0.0

    agent.decay_epsilon(1)
    """Dual Agent Systems"""
    agent_p.decay_epsilon(1)
    """END"""
    switch = True

    for timestep in range(total_steps):
        done = step(agent, agent_p, env)

        if done:
            if switch:
                if len(agent.memory) > 2500:
                    agent.train(2500)
                switch = not switch
            #"""Dual Agent Systems"""
            else:
                if len(agent_p.memory) > 2500:
                    agent_p.train(2500)
                switch = not switch
            """END"""


        if timestep % eval_frequency == 0:
            hours = (time() - start_time) / 3600

            """Single Model Call"""
            # sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
            #     comms, move, harvest = evaluate(agent, env, eval_episodes)
            """Dual Model Call"""
            sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
                comms, move, harvest = evaluate(agent, agent_p, env, eval_episodes)
            """END"""

            agent.update_target_from_model()
            """Dual Model"""
            agent_p.update_target_from_model()
            """END"""

            env.reset()

        print(
            f"Training Steps: {timestep}, Env: {env_str}, Sucess Rate: {sr:.2f}, Return: {ret:.2f}, Episode Length: {length:.2f}"
        )

    hours = (time() - start_time) / 3600

    """Single Model Call"""
    # sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
    #     comms, move, harvest = evaluate(agent, env, eval_episodes, True, env_str, start_time)
    """Dual Model Call"""
    sr, ret, length, avgAoI, peakAoI, dataDist, dataColl, CH_Metrics, \
        comms, move, harvest = evaluate(agent, agent_p, env, eval_episodes, True, env_str, start_time)
    """END"""


def step(agent, agent_p, env):
    """Single Model"""
    # train_model, train_CH, old_state, old_action, comms, move, harvest = env.step(agent)
    """Dual Model Call"""
    train_model, train_p, old_state, old_action, action_p, old_pstate, comms, move, harvest = env.step(agent, agent_p)
    """End"""

    buffer_done = env.terminated
    done = False

    if buffer_done or env.truncated:
        done = True

    if (train_model or env.truncated) and not buffer_done:
        print(f"Training")
        # QL/GANN
        # agent.update(old_state, old_action, env.archived_rewards,
        #              env.curr_state, buffer_done, env.curr_step)
        # DDQN
        agent.update_mem(old_state, old_action, env.archived_rewards,
                         env.curr_state, buffer_done, env.curr_step)

    """Dual Model Power Agent"""
    if train_p or done:
        agent_p.update_mem(old_pstate, int(action_p), env.archived_rewardsp, env.curr_pstate, buffer_done)
    """END"""

    return done


def prepopulate(agent, agent_p, prepop_steps, env, eval_frequency, lr):
    timestep = 0
    # QL
    agent.decay_epsilon(0)

    """Single Model"""
    # while timestep < prepop_steps:
    #     env.reset()
    #     done = False
    #
    #     while not done:
    #         print(f"Prepop Step: {timestep}, Reward: {env.full_reward}")
    #         train_model, train_CH, old_state, old_action, comms, move, harvest = env.step(agent)
    #         buffer_done = env.terminated
    #
    #         if buffer_done or env.truncated:
    #             done = True
    #
    #         if (train_model or env.truncated) and not buffer_done:
    #             agent.update_mem(old_state, old_action, env.archived_rewards,
    #                              env.curr_state, buffer_done, env.curr_step)
    #
    #             # QL/GANN Agents
    #             # agent.update(old_state, old_action, env.archived_rewards,
    #             #                  env.curr_state, buffer_done, env.curr_step)
    #         timestep += 1
    #
    #     if len(agent.memory) > 2048:
    #         agent.train(2048)
    #
    #     if timestep % eval_frequency == 0:
    #         # DDQN
    #         # for agent in agents:
    #         agent.update_target_from_model()
    #         env.reset()

    """Dual Model"""
    agent_p.decay_epsilon(0)
    agent.update_learning_rate((1 * lr) / prepop_steps)
    switch = True
    while timestep < prepop_steps:
        env.reset()
        done = False

        while not done:
            print(f"Prepop Step: {timestep}, Reward: {env.full_reward}")
            train_model, train_p, old_state, old_action, action_p, old_pstate, comms, move, harvest = (
                env.step(agent, agent_p))
            buffer_done = env.terminated

            if buffer_done or env.truncated:
                done = True

            if (train_model or env.truncated) and not buffer_done:
                agent.update_mem(old_state, old_action, env.archived_rewards,
                                 env.curr_state, buffer_done, env.curr_step)

            if train_p or done:
                agent_p.update_mem(old_pstate, int(action_p), env.archived_rewardsp, env.curr_pstate, buffer_done)

            timestep += 1

        if switch:
            if len(agent.memory) > 2500:
                agent.train(2500)
            switch = not switch
        else:
            if len(agent_p.memory) > 2500:
                agent_p.train(2500)
            switch = not switch

        if timestep % eval_frequency == 0:
            # DDQN
            # for agent in agents:
            agent.update_target_from_model()
            agent_p.update_target_From_model()
            env.reset()
            if timestep < 0.2 * prepop_steps:
                agent.update_learning_rate((timestep * lr) / (0.2 * prepop_steps))
            else:
                agent.update_learning_rate(lr)

def run_experiment(args):
    env_str = args.env
    print("Creating Evironment")
    env = UAV_IoT_Sim.make_env(scene=env_str, num_sensors=50, num_ch=5, num_uav=1, max_num_steps=720)

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    lr = 0.0001
    agent = model_utils.get_ddqn_agent(
        ((env.num_ch + 1) * 3),
        env.num_ch,
        alpha=lr,
        mem_len=25000
    )
    agent_p = model_utils.get_ddqn_agentp(
        env,
        4,
        10,
        alpha=lr,
        mem_len=25000
    )

    policy_save_dir = os.path.join(
        os.getcwd(), "policies", args.project_name
    )
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}"
    )

    prepopulate(agent, agent_p, 100_000, env, args.eval_frequency,lr)
    agent.update_learning_rate(lr)

    print("Beginning Training")
    train(
        agent,
        agent_p,
        env,
        args.env,
        args.steps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        # logger
    )


if __name__ == "__main__":
    run_experiment(get_args())
