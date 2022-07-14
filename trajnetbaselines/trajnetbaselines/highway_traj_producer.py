"""Command line tool to create a table of evaluations metrics."""
import datetime
import logging
import math
import os
import pdb
import pickle
import sys
import argparse
import numpy as np
import torch
import trajnettools

import gym
import highway_env
import matplotlib.pyplot as plt
from highway_env.vehicle.behavior import LinearVehicle
from highway_env.envs.common.action import action_factory
from rl_agents.agents.common.factory import agent_factory

from rl_agents.agents.common.factory import safe_deepcopy_env

import trajnetbaselines
from .scene_funcs.scene_funcs import scene_funcs

class TrajProducer(object):
    def __init__(self, file_name, sample_rate, start=0):  # nonlinear_scene_index
        self.file_name = file_name
        self.sample_rate = sample_rate
        self.start = start
        self.scene_funcs = scene_funcs()
        self.n_obs = None
        self.n_pred = None

        # The following part is for center line kd traj
        files = os.listdir("./center_lines/")
        self.center_line = {}
        for i in files:
            with open("./center_lines/" + i, "rb") as fp:  # Unpickling
                self.center_line[i] = torch.from_numpy(pickle.load(fp))
                # for j in range(0, len(self.center_line[i])):
                #     plt.plot(-self.center_line[i][j][0] / 8.5 + 984.7, self.center_line[i][j][1]/ 8.5 + 982.9)
                # plt.show()
        # This part is of no use if we use IDM and MOBIL but please keep it (otherwise the line calling the predictor, as well as the predictor itself needs to be modified)

    def aggregate(self, predictor, store_image=0):
        """
                store_image: if 1, means we want to store images of scene and predicted trajectories.
                """
        store_image_stride = 15  # sets how many images are going to be stored. every store_image_stride images, one image will be stored.
        scene_i = 0

        config = {
                "observation": {
                    "type": "KinematicsOrder",
                    "vehicles_count": 50,
                    "features": ["presence", "x", "y"],
                    "absolute": True,
                    "normalize": False,
                    "clip": False,
                    "order": "fixed",
                    "see_behind": True
                },
            }
        env = gym.make('dataset-merge-m-v0')
        env.verification_mode = True # Hide the ego vehicle
        env.PERCEPTION_DISTANCE = 1000
        env.configure(config)
        obs = env.reset()
        o = 1
        # print('initial obs:', obs)

        agent_config = {
            "__class__": "<class 'rl_agents.agents.tree_search.osla_simple_idm.OSLAIDMAgent'>",
            "env_preprocessors": [{"method": "simplify"}]
        }
        # agent = agent_factory(env, agent_config)
        real_obss = [[], [], [], []]
        # print(env.road.vehicles)
        for t in range(0, 4):
            v = 1
            real_obss[t].append([1, obs[0][1], obs[0][2]])
            while obs[v][0] == 1:
                real_obss[t].append([1, env.road.vehicles[v].obss[t][0], env.road.vehicles[v].obss[t][1]])
                v = v + 1
        simulation_length = 5

        # # xy[4] obs[5]
        # reference = [[1045.47, -960.98],
        #  [1044.62, -961.07],
        #  [1043.61, -961.19],
        #  [1042.41, -961.35],
        #  [1041.02, -961.54],
        #  [1039.52, -961.75],
        #  [1038.02, -961.97],
        #  [1036.54, -962.17],
        #  [1035.04, -962.38],
        #  [1033.45, -962.6],
        #  [1031.73, -962.85]]
        reference = env.reference
        # reference = 0

        ADE = 0
        FDE = 0
        num = 0

        for st in range(10):
            action = env.action_type.actions_indexes["IDLE"]
            obs, reward, done, info = env.step(action)
            real_obss.append(obs)

            # print('real obs:', obs[o])
            # if step > 3:
            #     print(obs)
            #     for v in range(0, len(real_obss[-1])):
            #         if real_obss[-1][v][0] == 0:
            #             for i in range(v + 1, len(real_obss[-1])):
            #                 real_obss[:][i - 1] = real_obss[:][i]
            if len(real_obss) > 5 and len(reference) > st + 1:
                print(obs[o], reference[st + 1])
                FDE = np.sqrt((obs[o][1] - reference[st + 1][0]) ** 2 + (obs[o][2] - reference[st + 1][1]) ** 2)
                print(FDE)
                # print('target speed and speed', env.road.vehicles[o].target_speed, env.road.vehicles[o].speed)
                num += 1
                print(num)
                ADE += FDE
            if st >= 0:
                env_copy = safe_deepcopy_env(env)  # Copy the env for x steps simulation
                simulate_obss = []  # To store the observation in the copied env
                frame = 0
                # print('copy start')
                for v in range(1, len(env_copy.road.vehicles)): # Except the MDP vehicle
                    env_copy.road.vehicles[v].desired_position = None
                while frame < simulation_length * 5:
                    action = env_copy.action_type.actions_indexes["IDLE"]
                    obs, reward, done, info = env_copy.step(action)
                    # env_copy.render()
                    simulate_obss.append(obs)
                    if frame == 0:
                        print(obs[o])
                    frame = frame + 5
                # print('copy end')

                # Use constant speed to expand simulate_obss to n_pred
                for t in range(simulation_length, self.n_pred):
                    simulate_obss.append(list())
                for i in range(0, len(simulate_obss[0])):
                    # speed_1 = (simulate_obss[simulation_length - 1][i][1] - simulate_obss[0][i][1]) / (
                    #             simulation_length - 1)
                    # speed_2 = (simulate_obss[simulation_length - 1][i][2] - simulate_obss[0][i][2]) / (
                    #             simulation_length - 1)
                    # for t in range(simulation_length, self.n_pred):
                    #     # print(simulate_obss)
                    #     simulate_obss[t].append(
                    #         [1, simulate_obss[simulation_length - 1][i][1] + speed_1 * (t - simulation_length),
                    #          simulate_obss[simulation_length - 1][i][2] + speed_2 * (t - simulation_length)])
                    if simulate_obss[simulation_length - 1][i][0] != 0:
                        speed_1 = (simulate_obss[simulation_length - 1][i][1] - simulate_obss[0][i][1]) / (simulation_length - 1)
                        speed_2 = (simulate_obss[simulation_length - 1][i][2] - simulate_obss[0][i][2]) / (simulation_length - 1)
                        for t in range(simulation_length, self.n_pred):
                            # print(simulate_obss)
                            simulate_obss[t].append(
                                [1, simulate_obss[simulation_length - 1][i][1] + speed_1 * (t - simulation_length),
                                simulate_obss[simulation_length - 1][i][2] + speed_2 * (t - simulation_length)])
                    # else:
                    #     speed_1 = (real_obss[-1][i][1] - real_obss[-simulation_length][i][1]) / (
                    #                 simulation_length - 1)
                    #     speed_2 = (real_obss[-1][i][2] - real_obss[-simulation_length][i][2]) / (
                    #                 simulation_length - 1)
                    #     for t in range(0, simulation_length):
                    #         # print(simulate_obss)
                    #         simulate_obss[t][i][:] = [1, real_obss[-1][i][1] + speed_1 * t,
                    #              real_obss[-1][i][2] + speed_2 * t]
                    #     for t in range(simulation_length, self.n_pred):
                    #         # print(simulate_obss)
                    #         simulate_obss[t].append(
                    #             [1, simulate_obss[simulation_length - 1][i][1] + speed_1 * (t - simulation_length),
                    #              simulate_obss[simulation_length - 1][i][2] + speed_2 * (t - simulation_length)])


                frames = np.linspace(self.start, self.start + 70, 15)
                # print('r:', real_obss)
                obss = real_obss[-5:] + simulate_obss[:]
                # print(obss)
                obs_converter = trajnetbaselines.ObsConverter()
                initial_paths = obs_converter.scene(obss, frames) # Paths before swapping order
                pixel_scale = torch.tensor([8.5])
                store_image_tmp = store_image * (int(scene_i % store_image_stride == 0))
                vehicle_number_used = 3 # numbers of vehicles used for prediction
                for v in range(0, len(obss[5 + simulation_length])):
                # for v in range(o, o + 1):
                    paths = initial_paths
                    # print(paths)
                    paths[0] = initial_paths[v]
                    paths[v] = initial_paths[0]
                    # print('last position:', paths[0])
                    if paths[0][4][2] == 0 and paths[0][4][3] == 0:
                        continue
                    prediction, test_time, scene_violation_smpl, my_flag = predictor(paths, n_obs=self.n_obs,
                                                                                 file_name='DR_CHN_Merging_ZS',
                                                                                 sample_rate=0.2,
                                                                                 pixel_scale=pixel_scale,
                                                                                 scene_funcs=self.scene_funcs,
                                                                                 store_image=store_image_tmp,
                                                                                 center_line=self.center_line)

                    # print('Pred:', prediction[0])
                    if v != 0:
                        # if v == o:
                        #     print('o predicted position', prediction[0][3], -prediction[0][2])
                        # print(env.road.vehicles[v].speed)
                        # print(initial_paths[0][5][3], -initial_paths[0][5][2])
                        env.road.vehicles[v].directly_set_position([prediction[0][3], -prediction[0][2]])
                        # # update the heading
                        # if prediction[0][3] != env.road.vehicles[v].position[0]:
                        #     env.road.vehicles[v].heading = np.pi + np.arctan((-prediction[0][2] - env.road.vehicles[v].position[1]) / (prediction[0][3] - env.road.vehicles[v].position[0]))
                    #         print('heading', env.road.vehicles[v].heading)
                        # env.road.vehicles[v].target_speed = 2 * np.sqrt((paths[v][-1][0] - paths[v][-2][0]) ** 2 + (paths[v][-1][1] - paths[v][-2][1]) ** 2)

            env.render()
        ADE = ADE / num
        print('ADE:', ADE)
        print('FDE:', FDE)
        self.result = [ADE, FDE]

def eval(input_file, predictor):
    # print('dataset', input_file)

    sample = 0.05 if 'syi.ndjson' in input_file else None
    file_name = []
    sample_rate = []

    # non-linear scenes from high Kalman Average L2
    n_obs = predictor.n_obs
    n_pred = predictor.n_pred

    producer = TrajProducer(file_name=file_name, sample_rate=sample_rate, start=0)  # nonlinear_scene_index
    producer.n_obs = n_obs  # setting n_obs and n_pred values
    producer.n_pred = n_pred


    producer.aggregate(predictor, store_image=0)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-add', default='output/rrb', type=str,
                        help='address to the model to be evaluated')
    parser.add_argument('--data-add', default='../trajnetdataset/output_interaction_sceneGeneralization/val/',
                        help='address to the test data')
    parser.add_argument('--image_add', default='output/',
                        help='address to the folder to store qualitative images')
    args = parser.parse_args()
    model = 'output/final_models/RRB/RRB_M_sceneGeneralization.pkl'
    predictor = trajnetbaselines.Predictor.load(model)
    list_data = os.listdir(args.data_add)
    test_files = [i for i in list_data]
    datasets = [args.data_add + i for i in test_files]
    results = {dataset
                   .replace('data/', '')
                   .replace('.ndjson', ''): eval(dataset, predictor)
               for i, dataset in enumerate(datasets)}


if __name__ == '__main__':
    main()








