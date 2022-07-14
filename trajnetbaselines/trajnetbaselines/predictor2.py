import numpy as np
import time
import pdb
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trajnettools
from casadi import *
from .utils import *
from . import augmentation
from .scene_funcs.scene_funcs import scene_preprocess
import matplotlib.pyplot as plt

visualize = 1
use_mpc = 1


class Predictor(object):
    def __init__(self, model, n_obs, n_pred, scale=1):
        self.model = model
        self.n_obs = n_obs
        self.n_pred = n_pred
        self.scale = scale

    def save(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # during development, good for compatibility across API changes:
        with open(filename + '.state_dict', 'wb') as f:
            torch.save(self.model.state_dict(), f)

    @staticmethod
    def load(filename):
        # torch.nn.Module.dump_patches = True
        with open(filename, 'rb') as f:
            return torch.load(f, map_location=torch.device('cpu'))

    def __call__(self, paths, n_obs=None, file_name=None, sample_rate=None, pixel_scale=None, scene_funcs=None,
                 store_image=0,
                 center_line=None):
        if (n_obs is None):
            n_obs = self.n_obs
        n_pred = self.n_pred
        start_time = time.time()
        self.model.eval()
        self.model.resampling_dim = (38, 74)
        # device = self.model.device
        device = 'cpu'
        self.model.to(device)
        ped_id = paths[0][0].pedestrian
        frame_diff = paths[0][1].frame - paths[0][0].frame
        first_frame = paths[0][n_obs].frame
        torch.backends.cudnn.enabled = False
        with torch.no_grad():

            xy = trajnettools.Reader.paths_to_xy(paths)
            xy = torch.tensor(xy, dtype=torch.float32, device=device).unsqueeze(0)
            scale = float(scene_funcs.pixel_scale_dict[file_name])
            offset = scene_funcs.offset_dict[file_name]
            xy_copy = xy.clone()
            print(offset)

            xy_copy[:, :, :, 1] = scale * (
                    xy[:, :, :, 1] - offset[0])  # second dimension is the longer axes, horizontal one
            xy_copy[:, :, :, 0] = scale * (xy[:, :, :, 0] - offset[1])
            for i in range(0, len(xy_copy)):
                for j in range(0, len(xy_copy[i])):
                    a = [point[0] for point in xy_copy[i][j]]
                    b = [point[1] for point in xy_copy[i][j]]
                    # plt.plot(a, b)
            xy_copy_unrotated = xy_copy.clone()
            rotated_scene, resampled_scene, xy_copy, theta = scene_preprocess(xy_copy, [file_name], n_obs,
                                                                              self.model.resampling_dim, scene_funcs)
            pixel_scale = pixel_scale.to(device)
            center_line_rotated = augmentation.rotate_all_path_by_theta(
                center_line[file_name + '.txt'].to(device=device), xy_copy[:, n_obs - 1:n_obs, 0:1], theta,
                centerline=1)

            '''for j in range(0, len(center_line_rotated[0])):
                plt.plot(center_line_rotated[0][j][0], center_line_rotated[0][j][1])
            plt.show()'''
            if 'RRB' in self.model.__class__.__name__:
                prediction_nn, prediction_kd, prob, prediction_rrb, prediction_speed = self.model(
                    obs=xy_copy[:, :n_obs, :, :], scene=resampled_scene,
                    sample_rate=torch.tensor([sample_rate], device=device), pixel_scale=pixel_scale,
                    center_line_dict=center_line_rotated, rotated_scene=rotated_scene[0], file_name=file_name,
                    margin=scene_funcs.return_margin([file_name]), kd = xy_copy[:, n_obs:, 0, :][0].tolist())
                print('kd pred:', prediction_kd)
                # print('rrb pred:', prediction_rrb)
                # print('nn pred:', prediction_nn)
                '''print('obs:', xy_copy[:, :n_obs, :, :])
                print('scene:', resampled_scene)
                print('sample rate:', torch.tensor([sample_rate], device=device))
                print('pixel scale:', pixel_scale)
                print('center line dict:', center_line_rotated)
                print('rotated scene:', rotated_scene[0])
                print('file name:', file_name)
                print('margin:', scene_funcs.return_margin([file_name]))'''
            elif 'EDN' in self.model.__class__.__name__:
                prediction_v, prediction_nn, prob, prediction_speed = self.model(obs=xy_copy[:, :n_obs, :, :],
                                                                                 scene=resampled_scene,
                                                                                 sample_rate=torch.tensor([sample_rate],
                                                                                                          device=device),
                                                                                 pixel_scale=pixel_scale,
                                                                                 center_line_dict=center_line_rotated,
                                                                                 rotated_scene=rotated_scene[0],
                                                                                 file_name=file_name,
                                                                                 margin=scene_funcs.return_margin(
                                                                                     [file_name]),
                                                                                 prediction_truth=xy_copy[:, n_obs:, :,
                                                                                                  :])

                prediction_kd = prediction_nn
                prediction_rrb = prediction_nn
            prob, prediction_nn, prediction_kd, prediction_rrb = prob[0], prediction_nn[0, :, :, :2].to(
                device), prediction_kd[0, :, :, :2].to(device), prediction_rrb[0, :, :, :2].to(
                device)  # as we have one batch
            prob = torch.nn.functional.softmax(prob, dim=0).cpu().numpy()
            # print("xy copy:", xy_copy[0, n_obs:])
            # print("prediction rrb:", prediction_rrb)
            best_mode_real_nn = l2_dist(xy_copy[0, n_obs:], prediction_nn)
            best_mode_real_rrb = l2_dist(xy_copy[0, n_obs:], prediction_rrb)
            best_mode_real_kd = l2_dist(xy_copy[0, n_obs:], prediction_kd)
            if (use_mpc):
                for i in range(prediction_rrb.shape[0]):
                    prediction_rrb[i] = mpc_fun(prediction_rrb[i], sample_rate, pixel_scale, n_pred,
                                                xy_copy[:, :n_obs, :, :])[1:]  # since the output has 11
            best_mode_prediction_nn = prediction_nn[0, :, :].reshape(-1, 2)  # torch.Size([n_pred, 2])
            best_mode_prediction_rrb = prediction_rrb[0, :, :].reshape(-1,
                                                                                        2)  # torch.Size([n_pred, 2])
            best_mode_prediction_kd = prediction_kd[0, :, :].reshape(-1, 2)  # torch.Size([n_pred, 2])
            best_mode_prediction_rrb_unrotated = best_mode_prediction_rrb.clone()
            best_mode_prediction_kd_unrotated = best_mode_prediction_kd.clone()
            best_mode_prediction_nn_unrotated = best_mode_prediction_nn.clone()
            prediction_nn_rotated = prediction_nn.clone()
            prediction_kd_rotated = prediction_kd.clone()
            prediction_rrb_rotated = prediction_rrb.clone()
            # print("best mode prediction rrb:", best_mode_prediction_rrb)

            rotation_enabled = 1  # if 1, it draws when we normalized the scene (for scene models) and if 0, draws without rotation.
            # print(theta)
            best_mode_prediction_nn = augmentation.rotate_path_by_theta(best_mode_prediction_nn,
                                                                        center=xy_copy[:, self.n_obs - 1, 0, :],
                                                                        n_pred=n_pred,
                                                                        theta=-theta)  # rotate back to original scene
            best_mode_prediction_kd = augmentation.rotate_path_by_theta(best_mode_prediction_kd,
                                                                        center=xy_copy[:, self.n_obs - 1, 0, :],
                                                                        n_pred=n_pred,
                                                                        theta=-theta)  # rotate back to original scene
            best_mode_prediction_rrb = augmentation.rotate_path_by_theta(best_mode_prediction_rrb,
                                                                         center=xy_copy[:, self.n_obs - 1, 0, :],
                                                                         n_pred=n_pred,
                                                                         theta=-theta)  # rotate back to original scene
            for i in range(prediction_nn.size(0)):  # num modes
                prediction_nn_rotated[i] = augmentation.rotate_path_by_theta(prediction_nn[i],
                                                                             center=xy_copy[:, self.n_obs - 1, 0,
                                                                                    :], n_pred=n_pred,
                                                                             theta=-theta)  # rotate back to original scene
                prediction_kd_rotated[i] = augmentation.rotate_path_by_theta(prediction_kd[i],
                                                                             center=xy_copy[:, self.n_obs - 1, 0,
                                                                                    :], n_pred=n_pred,
                                                                             theta=-theta)  # rotate back to original scene
                prediction_rrb_rotated[i] = augmentation.rotate_path_by_theta(prediction_rrb[i],
                                                                              center=xy_copy[:, self.n_obs - 1, 0,
                                                                                     :], n_pred=n_pred,
                                                                              theta=-theta)  # rotate back to original scene

            test_time = start_time - time.time()
            scene_violation = offroad_detector(prediction_rrb_rotated, file_name, scene_funcs.image, prob)

            projected_back_traj_rrb = best_mode_prediction_rrb.clone()
            projected_back_traj_rrb[:, 1] = best_mode_prediction_rrb[:, 1] / scale + offset[0]
            projected_back_traj_rrb[:, 0] = best_mode_prediction_rrb[:, 0] / scale + offset[1]

            projected_back_traj_nn = best_mode_prediction_nn.clone()
            projected_back_traj_nn[:, 1] = best_mode_prediction_nn[:, 1] / scale + offset[0]
            projected_back_traj_nn[:, 0] = best_mode_prediction_nn[:, 0] / scale + offset[1]

            projected_back_traj_kd = best_mode_prediction_kd.clone()
            projected_back_traj_kd[:, 1] = best_mode_prediction_kd[:, 1] / scale + offset[0]
            projected_back_traj_kd[:, 0] = best_mode_prediction_kd[:, 0] / scale + offset[1]

        flag = 0

        if ('Merging' in file_name):
            pass
        else:
            scene_violation_truth = offroad_detector(xy_copy_unrotated[:, 5:, 0], file_name, scene_funcs.image)
            if (scene_violation_truth > 1):  # and (error_kd_old-error_kd)>1):
                flag = 1
        if visualize:
            draw_scene(scene_funcs, xy_copy, file_name, rotated_scene, n_obs, n_pred, prob,
                       best_mode_prediction_rrb_unrotated.unsqueeze(0), rotation_enabled, ped_id,
                       first_frame, 'res', best_mode_prediction_nn_unrotated.unsqueeze(0),
                       best_mode_prediction_kd_unrotated.unsqueeze(0))
        return [trajnettools.TrackRow(first_frame + i * frame_diff, ped_id, x, y) for i, (x, y) in
                enumerate(projected_back_traj_rrb)], test_time, scene_violation, flag
