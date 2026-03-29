#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer import modified_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import Transform
import matplotlib.pyplot as plt
import json
import cv2


# code adapted from:
# https://github.com/abdo-eldesokey/pncnn/blob/c6122e9c442eabeb0145b241121aeba0039eb5e7/utils/sparsification_plot.py#L10
# https://github.com/poetrywanderer/CF-NeRF/blob/cbc9c3f8537bf404e8cbc19b55c63f1eb1271ab3/run_nerf_helpers.py#L382
# https://github.com/BayesRays/BayesRays/blob/edd549e323654c26d52797e43ef17de842befeef/bayesrays/metrics/ause.py#L6
def ause(unc_vec, err_vec, err_type='mae'):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0:int((1-r)*n_valid_pixels)]
        if err_type == 'rmse':
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae' or err_type == 'mse':
            ause_err.append(err_slice.mean().cpu().numpy())
       

    ###########################################

    # Sort by variance
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = []
    for r in ratio_removed:
        
        err_slice = err_vec_sorted_by_var[0:int((1 - r) * n_valid_pixels)]
        if err_type == 'rmse':
            ause_err_by_var.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == 'mae'or err_type == 'mse':
            ause_err_by_var.append(err_slice.mean().cpu().numpy())
    
    #Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)
    
    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause


def evaluate_uncertainty(dataset : ModelParams, iteration : int, pipeline : PipelineParams, viz : bool, gt_type : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        loaded_iter = scene.loaded_iter

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()

        transform = Transform(train_cameras[0].image_height, train_cameras[0].image_width)
        makedirs(os.path.join(dataset.model_path, "test", "ours_{}".format(loaded_iter)), exist_ok=True)
        if viz:
            uncertainties_path = os.path.join(dataset.model_path, "test", "ours_{}".format(loaded_iter), "visualisations", "uncertainty")
            plots_path = os.path.join(dataset.model_path, "test", "ours_{}".format(loaded_iter), "visualisations", "plots")
            makedirs(uncertainties_path, exist_ok=True)
            makedirs(plots_path, exist_ok=True)

        ause_metrics = []
        
        with torch.no_grad():
            train_render_imgs = []
            train_render_depths = []
            for train_idx, train_cam in enumerate(tqdm(train_cameras, desc="Rendering training set")):
                train_pkg = modified_render(train_cam, gaussians, pipeline, background)
                train_render_img = train_pkg["render"]   # 3 x H x W
                train_render_depth = train_pkg["depth"] # H x W
                train_render_img = F.interpolate(train_render_img.unsqueeze(0), (train_cameras[0].image_height, train_cameras[0].image_width), mode='bilinear').squeeze(0)
                train_render_depth = F.interpolate(train_render_depth.unsqueeze(0).unsqueeze(0), (train_cameras[0].image_height, train_cameras[0].image_width), mode='nearest').squeeze(0).squeeze(0)
                train_render_imgs.append(train_render_img.cpu())
                train_render_depths.append(train_render_depth.cpu())

            for test_idx, test_cam in enumerate(tqdm(test_cameras, desc="Evaluating AUSE on test set")):
                test_pkg = modified_render(test_cam, gaussians, pipeline, background)
                test_render_img = test_pkg["render"]   # 3 x H x W
                test_render_depth = test_pkg["depth"] # H x W
                test_render_img = F.interpolate(test_render_img.unsqueeze(0), (train_cameras[0].image_height, train_cameras[1].image_width), mode='bilinear').squeeze(0)
                test_render_depth = F.interpolate(test_render_depth.unsqueeze(0).unsqueeze(0), (train_cameras[0].image_height, train_cameras[1].image_width), mode='nearest').squeeze(0).squeeze(0)
                test_gt_img = test_cam.original_image[0:3, :, :]
                if gt_type == "scannetpp":
                    test_gt_depth = cv2.imread(os.path.join(dataset.source_path, 'depths', test_cam.image_name+'.png'), cv2.IMREAD_UNCHANGED) / 1000
                elif gt_type == "ETH3D":
                    with open(os.path.join(dataset.source_path, 'depths', test_cam.image_name+'.npy'), 'rb') as f:
                        test_gt_depth = np.load(f)

                test_gt_depth = torch.from_numpy(test_gt_depth).float().to("cuda")
                mask = (test_gt_depth > 0) & (test_gt_depth != torch.inf)
                masked_gt_depth = test_gt_depth[mask]
                resized_render_depth = F.interpolate(test_render_depth.unsqueeze(0).unsqueeze(0), (test_gt_depth.shape[0], test_gt_depth.shape[1]), mode='nearest').squeeze(0).squeeze(0)
                masked_rendered_depth = resized_render_depth[mask]
                err_vec = (abs(masked_gt_depth - masked_rendered_depth)).flatten()
                mae = (test_gt_depth - resized_render_depth).abs()
                mae[~mask] = 0
                
                point_diffs = []
                vis_masks = []
                
                for train_idx, train_cam in enumerate(train_cameras):
                    train_render_img = train_render_imgs[train_idx].to("cuda")
                    train_render_depth = train_render_depths[train_idx].to("cuda")
                    test_points = transform.backproject(test_render_depth, test_cam.K)
                    test_points_transformed = transform.transform_T(test_points, train_cam.world_view_transform.T @ test_cam.world_view_transform.T.inverse())
                    source_pixels = transform.project(test_points_transformed, train_cam.K)
                    warped_img = transform.sample(train_render_img, source_pixels)
                    train_points = transform.backproject(train_render_depth, train_cam.K)
                    train_points_transformed = transform.transform_T(train_points, test_cam.world_view_transform.T @ train_cam.world_view_transform.T.inverse())
                    corresponding_train_points = transform.sample(train_points_transformed, source_pixels, mode='nearest')
                    
                    positive_depth = (test_points_transformed[2]>0)
                    vis_mask = positive_depth
                    point_diff = (corresponding_train_points - test_points) * vis_mask
                    point_diffs.append(point_diff)
                    vis_masks.append(vis_mask)
                    

                point_diffs = torch.stack(point_diffs, dim=0)
                vis_mask = torch.stack(vis_masks, dim=0)
                uncertainty = (point_diffs[:,2].abs().sum(0)/vis_mask.sum(0)) # mean abs depth diff
                # AUSE
                clean_max = uncertainty[~uncertainty.isnan()].max()
                uncertainty[uncertainty.isnan()] = clean_max
                uncertainty = F.interpolate(uncertainty.unsqueeze(0).unsqueeze(0), (test_gt_depth.shape[0], test_gt_depth.shape[1]), mode='nearest').squeeze(0).squeeze(0)
                unc_vec = uncertainty[mask].flatten()
                ratio_removed, ause_err, ause_err_by_var, ause_metric = ause(unc_vec, err_vec)
                print(ause_metric)
                ause_metrics.append(ause_metric)
                if viz:
                    plot_errors(ratio_removed, ause_err, ause_err_by_var, os.path.join(plots_path, '{0:05d}'.format(test_idx) + ".jpg"))
                    plt.imsave(os.path.join(uncertainties_path, f'{test_idx:05d}_{test_cam.image_name}' ".jpg"), uncertainty.abs().cpu().numpy(), vmin=unc_vec.quantile(0.01), vmax=unc_vec.quantile(0.99))
                
            ause_metrics = np.array(ause_metrics)
            print("Average AUSE: ", np.mean(ause_metrics))
            with open(os.path.join(dataset.model_path, "test", "ours_{}".format(loaded_iter), "ause.json"), "w") as f:
                json.dump({'ause mean': np.mean(ause_metrics),
                           'all ause': ause_metrics.tolist()}, f, ensure_ascii=False, indent=4)

                
                


def plot_errors(ratio_removed, ause_err, ause_err_by_var, output_path): #AUSE plots, with oracle curve also visible
    plt.plot(ratio_removed, ause_err, '--')
    plt.plot(ratio_removed, ause_err_by_var, '-r')
    plt.plot(ratio_removed, ause_err_by_var - ause_err, '-g') # uncomment for getting plots similar to the paper, without visible oracle curve
    plt.savefig(output_path)
    plt.close()
    plt.figure()




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--gtdepth", type=str, default="scannetpp")
    args = get_combined_args(parser)
    print("Evaluating uncertainty for " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    evaluate_uncertainty(model.extract(args), args.iteration, pipeline.extract(args), args.viz, args.gtdepth)