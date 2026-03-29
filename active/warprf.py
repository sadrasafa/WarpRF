import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Union, Optional
from copy import deepcopy
import random
from gaussian_renderer import render, network_gui, modified_render
from scene import Scene
from utils.graphics_utils import Transform


class WarpRFSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()

    
    def nbvs(self, gaussians, scene: Scene, num_views, pipe, background, exit_func) -> List[int]:
        candidate_views = deepcopy(list(scene.get_candidate_set()))
        
        candidate_cameras = scene.getCandidateCameras()
        cam_uncertainties = torch.zeros(len(candidate_cameras))

        train_cameras = scene.getTrainCameras().copy()
        transform = Transform(train_cameras[0].image_height, train_cameras[0].image_width)

        with torch.no_grad():
            for idx, cam in enumerate(tqdm(candidate_cameras, desc="Calculating uncertainty on candidate views")):
                if exit_func():
                    raise RuntimeError("csm should exit early")
                candidate_pkg = modified_render(cam, gaussians, pipe, background)
                candidate_img = candidate_pkg["render"]   # 3 x H x W
                candidate_depth = candidate_pkg["depth"] # H x W
                train2candidates = []
                for train_idx, train_cam in enumerate(train_cameras):
                    train_img = modified_render(train_cam, gaussians, pipe, background)["render"]
                    warped = transform.warp_source2target(train_img, train_cam, cam, candidate_depth)
                    train2candidates.append(warped)
                diff = torch.stack(train2candidates, dim=0) - candidate_img # num_train_views x 3 x H x W
                metric = diff.abs().sum(1)  # num_train_views x H x W
                min_metric, _ = metric.min(0) # H x W
                uncertainty = min_metric.sum() # scalar
                cam_uncertainties[idx] = uncertainty
        _, indices = torch.sort(cam_uncertainties, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        return selected_idxs
    
    
    def forward(self, x):
        return x