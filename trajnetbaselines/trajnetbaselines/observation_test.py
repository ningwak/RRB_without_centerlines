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

import trajnetbaselines
from .scene_funcs.scene_funcs import scene_funcs
from rl_agents.agents.common.factory import agent_factory

class Evaluator(object):
    def __init__(self, file_name, sample_rate):  # nonlinear_scene_index
        self.file_name = file_name
        self.sample_rate = sample_rate
        self.scene_funcs = scene_funcs()
        self.n_obs = None
        self.n_pred = None

        self.center_line = {}
        i = 'DR_CHN_Merging_ZS.txt'
        with open("./center_lines/" + i, "rb") as fp:  # Unpickling
            self.center_line[i] = torch.from_numpy(pickle.load(fp))

    def aggregate(self, predictor, store_image=0):
        """
        store_image: if 1, means we want to store images of scene and predicted trajectories.
        """
        store_image_stride = 15  # sets how many images are going to be stored. every store_image_stride images, one image will be stored.
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
        scene_i = 0
        if True:
            '''for _, _ in enumerate(
                self.scenes):'''  # paths is a list of the pedestrian of interest and other neighbors
            pixel_scale = torch.tensor([float(self.scene_funcs.pixel_scale_dict[self.file_name[scene_i]])])
            store_image_tmp = store_image * (int(scene_i % store_image_stride == 0))

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

            '''agent_config = {
                "__class__": "<class 'rl_agents.agents.tree_search.brue.BRUEAgent'>",
                "env_preprocessors": [{"method": "simplify"}]
            }
            agent = agent_factory(env, agent_config)'''

            frame = 0
            obs_converter = trajnetbaselines.ObsConverter()
            done = False
            obss = []
            '''for _ in range(15):
                # action = agent.act(obs)
                action = env.action_type.actions_indexes["IDLE"]
                obs, reward, done, info = env.step(action)
                env.render()'''
            while frame <= 70:
                frame = frame + 5
                '''while i < 5:
                    i = i + 1'''
                # action = env.action_space.sample()
                # action = agent.act(obs)
                action = env.action_type.actions_indexes["IDLE"]
                obs, reward, done, info = env.step(action)
                env.render()
                obss.append(obs)
                # print(len(obs))
                # print(obs[1])
            env.close()
            frames = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
            # print('obss:', obss)
            paths_before = obs_converter.scene(obss, frames)
            paths = paths_before[1:]
            '''temp = paths[0]
            paths[0] = paths[8]
            paths[8] = temp'''
            print('paths:', paths)
            self.scenes = paths
            '''print("n_obs:", self.n_obs)
            print("pixel_scale:", pixel_scale)'''
            # print('file name:', self.file_name)
            # print(self.center_line)

            prediction, test_time, scene_violation_smpl, my_flag = predictor(paths, n_obs=self.n_obs,
                                                                             file_name=self.file_name[scene_i],
                                                                             sample_rate=self.sample_rate[scene_i],
                                                                             pixel_scale=pixel_scale,
                                                                             scene_funcs=self.scene_funcs,
                                                                             store_image=store_image_tmp,
                                                                             center_line=self.center_line)
            print('Pred:', prediction)


            allPredictions.append(prediction)
        plt.show()
        self.allPredictions = allPredictions
        return

def eval(input_file, predictor):
    print('dataset', input_file)

    sample = 0.05 if 'syi.ndjson' in input_file else None
    reader = trajnettools.Reader(input_file, scene_type='paths')
    file_name = []
    sample_rate = []
    scene_instance = scene_funcs(device='cpu').to('cpu')
    for files, idx, s, rate in reader.scenes(sample=sample):
        file_name.append(files)
        sample_rate.append(rate)
    del scene_instance

    # non-linear scenes from high Kalman Average L2
    n_obs = predictor.n_obs
    n_pred = predictor.n_pred

    evaluator = Evaluator(file_name=file_name, sample_rate=sample_rate)  # nonlinear_scene_index
    evaluator.n_obs = n_obs  # setting n_obs and n_pred values
    evaluator.n_pred = n_pred

    evaluator.aggregate(predictor, store_image=0)
    return evaluator.allPredictions


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

