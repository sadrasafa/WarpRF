# WarpRF: Multi-View Consistency for Training-Free Uncertainty Quantification and Applications in Radiance Fields

### [Project Page](https://kuis-ai.github.io/WarpRF/) | [Paper](https://arxiv.org/abs/2506.22433)

Official implementation of: "**WarpRF: Multi-View Consistency for Training-Free Uncertainty Quantification and Applications in Radiance Fields**"
_Accepted at_ [_WACV 2026_](https://wacv.thecvf.com/)

by [Sadra Safadoust](https://sadrasafa.github.io/), [Fabio Tosi](https://fabiotosi92.github.io/), [Fatma Güney](https://mysite.ku.edu.tr/fguney/), and [Matteo Poggi](https://mattpoggi.github.io/)



## :bookmark_tabs: Table of Contents

1. [Overview](#watermelon-overview)
2. [Installation](#gear-installation)
3. [Active View Selection](#camera_flash-active-view-selection)
4. [Uncertainty Quantification](#chart_with_downwards_trend-uncertainty-quantification) 
5. [Citation](#fountain_pen-citation)
6. [Acknowledgements](#pray-acknowledgements)

## :watermelon: Overview

WarpRF introduces a training-free uncertainty quantification framework for radiance fields by leveraging multi-view consistency. It does not require modifiying the internal strucutre of the radiance fields, making it compatible with a wide range of representations such as 3D Gaussian Splatting (3DGS), NeRF, SVRaster, and more. Additionally, it requires no changes to the training procedure, allowing it to be applied directly to trained radiance fields. This repository provides the implementation for 3DGS experiments, with code for NeRF and SVRaster experiments coming soon.

## :gear: Installation
Clone our repository with submodules:

```bash
git clone git@github.com:sadrasafa/WarpRF.git --recursive
```
Create a conda environment and install the requirements:
```bash
conda create -n warprf python=3.10
conda activate warprf
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-build-isolation submodules/diff-gaussian-rasterization/
pip install --no-build-isolation submodules/simple-knn/
pip install --no-build-isolation -e ./diff/ -v
```

## :camera_flash: Active-View Selection

Download the [MipNeRF360](https://jonbarron.info/mipnerf360/) and [NeRF-Synthetic](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset) datasets and place them in your preferred data directory.
Use the `scripts/active_*.sh` files to run the active view selection with 3DGS experiments. 
```bash
# active view selection on MipNeRF360
bash scripts/active_mipnerf.sh   [PATH_TO_SCENE] [OUTPUT_DIR] 

# active view selection on NeRF-Synthetic with 20 views
bash scripts/active_blender20.sh [PATH_TO_SCENE] [OUTPUT_DIR] 

# active view selection on NeRF-Synthetic with 10 views
bash scripts/active_blender10.sh [PATH_TO_SCENE] [OUTPUT_DIR] 
```


## :chart_with_downwards_trend: Uncertainty Quantification

### ETH3D
Download the [ETH3D](https://www.eth3d.net/datasets). We only need the undistorted jpg images and the undistorted depths. Note that only ground-truth depth that match the distorted images are provided, therefore they need to be undistorted given the camera parameters. (see [here](https://github.com/sadrasafa/StereoGS/issues/4#issuecomment-3343345542)). To match our train/val split, download and unzip the split files from [here](https://drive.google.com/file/d/13RnaOgueWMVLkm5W2jpxp6tjWK3zmSZV/view?usp=sharing).   

### ScanNet++

Download the [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) dataset. We use the DSLR data for the following scenes:
`27dd4da69e`,  `3864514494`,  `5eb31827b7`,  `8b5caf3398`,  `8d563fc2cc`,  `b20a261fdf`.
Follow the instructions at [Official ScanNet++ Toolkit](https://github.com/scannetpp/scannetpp) to render depth and then undistort the images and depths. Note that the provided undistortion script only undistorts the images, however it can be easily extended to undistort depths too (e.g., check [this](https://github.com/scannetpp/scannetpp/issues/65#issuecomment-1939346286)). Also, it saves the camera intrinsics for the undistorted pinhole camera in the nerfstudio's json format, so make sure to update the camera parameters in the colmap format (`cameras.txt`) accordingly as well. To match our train/val split, download and unzip the split files from [here](https://drive.google.com/file/d/13RnaOgueWMVLkm5W2jpxp6tjWK3zmSZV/view?usp=sharing).


<details>

<summary>Expected Directory Structure</summary>

At the end you should have the following directory structure:
```
├── [DATASET-NAME]
    ├── [SCENE-NAME]
        ├── split.json
        ├── images
        ├── depths
        ├── sparse
            ├── 0
                ├── cameras.txt
                ├── images.txt
                ├── points3D.txt
```
</details>

<br>

Use the `scripts/ause_*.sh` files to run uncertainty quantification with 3DGS experiments. 
```bash
# uncertainty quantification on ETH3D
bash scripts/ause_ETH3D.sh     [PATH_TO_SCENE] [OUTPUT_DIR]

 # uncertainty quantification on ScanNet++
bash scripts/ause_scannetpp.sh [PATH_TO_SCENE] [OUTPUT_DIR]
```


## :fountain_pen: Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{safadoust2026warprf,
  title={WarpRF: Multi-View Consistency for Training-Free Uncertainty Quantification and Applications in Radiance Fields},
  author={Safadoust, Sadra and Tosi, Fabio and G{\"u}ney, Fatma and Poggi, Matteo},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2026}
}
```

## :pray: Acknowledgements
This project builds heavily on [FisherRF](https://github.com/JiangWenPL/FisherRF) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).  We are grateful to the authors for making their code publicly available.
