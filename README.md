<!-- Use this for pypi package (and disable above). Hacky workaround -->
<!-- <p align="center">
    <img alt="nerfstudio" src="https://docs.nerf.studio/en/latest/_images/logo.png" width="400">
</p> 

<p align="center"> A collaboration friendly studio for NeRFs </p> -->

<!-- 
<img src="https://user-images.githubusercontent.com/3310961/194017985-ade69503-9d68-46a2-b518-2db1a012f090.gif" width="52%"/> <img src="https://user-images.githubusercontent.com/3310961/194020648-7e5f380c-15ca-461d-8c1c-20beb586defe.gif" width="46%"/>

- [Quickstart](#quickstart)
- [Learn more](#learn-more)
- [Supported Features](#supported-features)

-->

# About

Mr. NeSLAM is a NeRF SLAM algorithm intended for mobile robots based on NerfStudio.

## 1. Installation: Setup the environment

### Prerequisites

CUDA must be installed on the system. This library has been tested with version 11.3. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```

### Dependencies

Install pytorch with CUDA (this repo has been tested with CUDA 11.3) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Installing nerfstudio

Easy option:

```bash
pip install nerfstudio
```

If you would want the latest and greatest:

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

## 2. Setting up the data

Download the original NeRF Blender dataset. We support the major datasets and allow users to create their own dataset, described in detail [here](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html).

```bash
ns-download-data --dataset=blender
ns-download-data --dataset=nerfstudio --capture=poster
```

### 2.x Using custom data

If you have custom data in the form of a video or folder of images, we've provided some [COLMAP](https://colmap.github.io/) and [FFmpeg](https://ffmpeg.org/download.html) scripts to help you process your data so it is compatible with nerfstudio.

After installing both software, you can process your data via:

```bash
ns-process-data {video,images,insta360} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
# Or if you're on a system without an attached display (i.e. colab):
ns-process-data {video,images,insta360} --data {DATA_PATH}  --output-dir {PROCESSED_DATA_DIR} --no-gpu
```

## 3. Training a model

To run with all the defaults, e.g., vanilla NeRF method with the Blender Lego image

```bash
# To see what models are available.
ns-train --help

# To see what model-specific cli arguments are available.
ns-train nerfacto --help

# Run with nerfacto model.
ns-train nerfacto

# We provide support for other models. E.g., to run instant-ngp.
ns-train instant-ngp

# To train on your custom data.
ns-train nerfacto --data {PROCESSED_DATA_DIR}
```

### 3.x Training a model with the viewer

You can visualize training in real-time using our web-based viewer.

Make sure to forward a port for the websocket to localhost. The default port is 7007, which you should expose to localhost:7007.

```bash
# with the default port
ns-train nerfacto --vis viewer

# with a specified websocket port
ns-train nerfacto --vis viewer --viewer.websocket-port=7008

# port forward if running on remote
ssh -L localhost:7008:localhost:7008 {REMOTE HOST}
```

For more details on how to interact with the visualizer, please visit our viewer [walk-through](https://docs.nerf.studio/en/latest/quickstart/viewer_quickstart.html).

## 4. Rendering a trajectory during inference

After your model has trained, you can headlessly render out a video of the scene with a pre-defined trajectory.

```bash
# assuming previously ran `ns-train nerfacto`
ns-render --load-config=outputs/data-nerfstudio-poster/nerfacto/{TIMESTAMP}/config.yml --traj=spiral --output-path=renders/output.mp4
```
