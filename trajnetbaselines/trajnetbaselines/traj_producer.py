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

from rl_agents.agents.common.factory import safe_deepcopy_env

import trajnetbaselines
from .scene_funcs.scene_funcs import scene_funcs


class Producer(object):
    def __init__(self, scenes, file_name, sample_rate, start):  # nonlinear_scene_index
        self.scenes = scenes
        self.file_name = file_name
        self.sample_rate = sample_rate
        self.start = start
        self.average_l2 = {'N': len(scenes)}
        self.final_l2 = {'N': len(scenes)}
        self.cross_track = {'N': len(scenes)}
        self.scene_violations = {'N': len(scenes)}
        self.variance_error = {'N': len(scenes)}
        self.avg_test_time = {'N': len(scenes)}
        self.scene_funcs = scene_funcs()
        self.n_obs = None
        self.n_pred = None


        # The following part is for center line kd traj
        files = os.listdir("./center_lines/")
        self.center_line = {}
        for i in files:
            with open("./center_lines/" + i, "rb") as fp:  # Unpickling
                self.center_line[i] = torch.from_numpy(pickle.load(fp))
        #This part is of no use if we use IDM and MOBIL but please keep it (otherwise the line calling the predictor, as well as the predictor itself needs to be modified)

    def aggregate(self, predictor, store_image=0):
        """
        store_image: if 1, means we want to store images of scene and predicted trajectories.
        """
        store_image_stride = 15  # sets how many images are going to be stored. every store_image_stride images, one image will be stored.
        scene_i = 0

        average = 0.0
        nonlinear = 0.0
        final = 0.0
        cross_track = 0.0
        scene_violation = 0.0
        allPredictions = []
        tot_test_time = 0
        l2_error = []
        flag = 1
        cnt = 0

        # The following part is for IDM+MOBIL
        config = {
            "observation": {
                "type": "KinematicsOrder",
                "vehicles_count": 13,
                "features": ["presence", "x", "y"],
                "absolute": True,
                "normalize": False,
                "clip": False,
                "order": "sorted",
                "see_behind": True
            }
        }
        env = gym.make('dataset-merge-m-v0')
        env.configure(config)
        obs = env.reset()
        env.action_type = action_factory(env, {"type": "ContinuousAction"})
        # # state.action_type.clip = False
        # idm_ego = LinearVehicle.create_from(env.road.vehicles[0])

        frame = self.start
        obs_converter = trajnetbaselines.ObsConverter()
        done = False
        real_obss = [] # To store the real observation
        env_copy = safe_deepcopy_env(env) # Copy the env for x steps simulation
        simulation_length = 3
        simulate_obss = [] # To store the observation in the copied env
        while frame <= self.start + simulation_length * 5:
            action = [0, 0]
            obs, reward, done, info = env_copy.step(action)
            env_copy.render()
            if frame != self.start:
                simulate_obss.append(obs)
            frame = frame + 5
        # print('simulate_obss:', simulate_obss)

        # Use constant speed to expand simulate_obss to n_pred
        for t in range(simulation_length, self.n_pred):
            simulate_obss.append([])
        for i in range(0, len(simulate_obss[0])):
            speed_1 = (simulate_obss[simulation_length - 1][i][1] - simulate_obss[0][i][1]) / (simulation_length - 1)
            speed_2 = (simulate_obss[simulation_length - 1][i][2] - simulate_obss[0][i][2]) / (simulation_length - 1)
            for t in range(simulation_length, self.n_pred):
                simulate_obss[t].append([1, simulate_obss[simulation_length - 1][i][1] + speed_1 * (t - simulation_length),
                                         simulate_obss[simulation_length - 1][i][2] + speed_2 * (t - simulation_length)])
        print('simulate_obss:', simulate_obss)
        obss = simulate_obss[0:10]

        frames = np.linspace(self.start, self.start + 70, 10)
        paths_before = obs_converter.scene(obss, frames)
        idm_path = paths_before[1:]
        # IDM+MOBIL part end

        for scene_i, paths in enumerate(
                self.scenes):  # paths is a list of the pedestrian of interest and other neighbors
            dataset_path = paths[0].copy()
            paths[:][5:] = idm_path # comment this line if use center_line
            # print('Paths:', paths)
            pixel_scale = torch.tensor([float(self.scene_funcs.pixel_scale_dict[self.file_name[scene_i]])])
            store_image_tmp = store_image * (int(scene_i % store_image_stride == 0))

            prediction, test_time, scene_violation_smpl, my_flag = predictor(paths, n_obs=self.n_obs,
                                                                             file_name=self.file_name[scene_i],
                                                                             sample_rate=self.sample_rate[scene_i],
                                                                             pixel_scale=pixel_scale,
                                                                             scene_funcs=self.scene_funcs,
                                                                             store_image=store_image_tmp,
                                                                             center_line=self.center_line)

            print('Pred:', prediction)
            allPredictions.append(prediction)
            average_l2 = trajnettools.metrics.average_l2(dataset_path, prediction, self.n_pred)
            final_l2 = trajnettools.metrics.final_l2(dataset_path, prediction)
            cross_track_l2 = trajnettools.metrics.cross_track(dataset_path, prediction, self.n_pred)

            # aggregate

            if (my_flag):
                scene_violation -= scene_violation_smpl
                cnt += 1
            average += average_l2
            scene_violation += scene_violation_smpl
            l2_error.append(average_l2)
            final += final_l2
            cross_track += cross_track_l2
            tot_test_time += test_time

        average /= len(self.scenes) - cnt

        print(cnt, ", cnt normalized", cnt / len(self.scenes))
        final /= len(self.scenes)
        cross_track /= len(self.scenes)
        scene_violation /= len(self.scenes)
        l2_error_pow2 = [(i - average) ** 2 for i in l2_error]
        self.average_l2['model'] = average
        self.final_l2['model'] = final
        self.cross_track['model'] = cross_track
        self.scene_violations['model'] = scene_violation
        self.variance_error['model'] = math.sqrt(sum(l2_error_pow2) / len(l2_error_pow2))
        self.avg_test_time['model'] = tot_test_time / len(self.scenes)
        return

    def result(self):
        return self.average_l2, self.final_l2, self.variance_error, self.avg_test_time, self.scene_violations, self.cross_track  # self.average_l2_nonlinear,


def eval(input_file, predictor):
    print('dataset', input_file)

    sample = 0.05 if 'syi.ndjson' in input_file else None
    reader = trajnettools.Reader(input_file, scene_id=38928, scene_type='paths')
    file_name = []
    scenes = []
    sample_rate = []
    scene_instance = scene_funcs(device='cpu').to('cpu')
    for files, idx, s, rate in reader.scenes(sample=sample):
        scenes.append(s)
        file_name.append(files)
        sample_rate.append(rate)
    del scene_instance

    # non-linear scenes from high Kalman Average L2
    n_obs = predictor.n_obs
    n_pred = predictor.n_pred

    evaluator = Producer(scenes, file_name=file_name, sample_rate=sample_rate, start = reader.start)  # nonlinear_scene_index
    evaluator.n_obs = n_obs  # setting n_obs and n_pred values
    evaluator.n_pred = n_pred

    evaluator.aggregate(predictor, store_image=0)
    return evaluator.result()


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

    ADE = np.zeros([3])
    FDE = np.zeros([3])
    Num = np.zeros([3])
    Scene_viol = np.zeros([3])
    cross_track = np.zeros([3])
    for i in datasets:
        N = results[i[:-7]][0]['N']
        if ('Roundabout' in i):
            ADE[0] += results[i[:-7]][0]['model'] * N
            FDE[0] += results[i[:-7]][1]['model'] * N
            Num[0] += N
            Scene_viol[0] += results[i[:-7]][-2]['model'] * N
            cross_track[0] += results[i[:-7]][-1]['model'] * N
        elif ('Intersection' in i):
            ADE[1] += results[i[:-7]][0]['model'] * N
            FDE[1] += results[i[:-7]][1]['model'] * N
            Num[1] += N
            Scene_viol[1] += results[i[:-7]][-2]['model'] * N
            cross_track[1] += results[i[:-7]][-1]['model'] * N
        elif ('Merging' in i):
            ADE[2] += results[i[:-7]][0]['model'] * N
            FDE[2] += results[i[:-7]][1]['model'] * N
            Num[2] += N
            Scene_viol[2] += results[i[:-7]][-2]['model'] * N
            cross_track[2] += results[i[:-7]][-1]['model'] * N

    ADE = ADE / Num
    FDE = FDE / Num
    Scene_viol = Scene_viol / Num
    cross_track = cross_track / Num
    print('Roundabout', 'Intersection', 'Merging')
    print(ADE)
    print(FDE)
    print(Scene_viol)
    print(cross_track)
    print('average values:')
    print(np.sum(ADE) / 3)
    print(np.sum(FDE) / 3)
    print(np.sum(Scene_viol) / 3)
    print(np.sum(cross_track) / 3)


if __name__ == '__main__':
    main()
