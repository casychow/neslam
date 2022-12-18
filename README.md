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

# NeSLAM: NeRF and Visual Simultaneous Localization and Mapping
NeSLAM is a NeRF SLAM algorithm intended for mobile robots created by Cassandra Chow and Diar Sanakov. The algorithm is based on NerfStudio. This is the class project for the ROB 6203: Robot Perception course at New York University.

A table of contents has been provided for ease-of-project access. The final video listed below narrates how the project went and what our results were. The following README.md document is to inform others how we setup our environments, if anyone would like to recreate them. If so, please do share and give credit. Thank you.

# Table of Contents
- Results
- Getting Started
- Phase 1: COLMAP, FFmpeg, and NeRFStudio
    - Setup environment for COLMAP, FFmpeg, and NeRFStudio
        - Setting up FFmpeg
        - Setting up COLMAP
        - Setting up NeRFStudio
        - Running NeRFStudio
    - Training a model
- Phase 2: ORB-SLAM
    - Setting up
    - Running ORB-SLAM3 with camera and custom data


# Results
<!-- [NeSLAM Project Video](https://youtu.be/jSPsX-cWzDQ) -->
[![Watch the video](https://i.imgur.com/bhhHcS6.jpg)](https://youtu.be/jSPsX-cWzDQ)



# Getting Started
If you are getting started with Windows, you will need to download the following
* Git
* Windows 10 (we did not test using Windows 7)
* Conda
* CUDA version >= 1.11.3
* ROS

# Phase 1: COLMAP, FFmpeg, and NeRFStudio
FFmpeg is used to generally process audio and video files. A more specific use case we will be using FFmpeg for is breaking up a video file into images. COLMAP will then use each image to find its the camera pose. NeRFStudio, an API implementation of Neural Radiance Fields (NeRF), will take COLMAP's output of images with camera poses and recreate a rendering of the video.

## Setup environment for COLMAP, FFmpeg, and NeRFStudio

### Setting up FFmpeg (Windows)
1. Open a command prompt to a custom directory named "ffmpeg" in the C:\ drive
```
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
```
2. Edit local system environment variables and include a new entry for the 'PATH' variable as: 'C:\ffmpeg\bin'.

Or, follow this video's instructions: https://www.youtube.com/watch?v=r1AtmY-RMyQ.

### Setting up COLMAP (Windows)
1. Open a command prompt to a directory you will use FFmpeg, COLMAP, and NeRFStudio.
```
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install colmap[cuda]:x64-windows
```
To install packets in general, or in x64
```
.\vcpkg install [packages to install]
.\vcpkg install [package name]:x64-windows
```
Note: If you experience a triplet x64-windows error, repair Visual Studio through the Visual Studio Installer.

### Running FFmpeg and COLMAP
Run the following command. You can only choose one choice from the curly brackets. If you would like to run this command only using your cpu, add "--no-gpu" at the end of the command.
```
ns-process-data {video,images,insta360} --data DATA_PATH --output-dir PROCESSED_DATA_DIR
```

### Setting up NeRFStudio (Windows)
0. Install CUDA and conda
1. Create environment (taken from NeRFStudio's website)
```
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```
2. Dependencies
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
3. Installing nerfstudio

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
4. Using data

NeRF Blender dataset: Download the original NeRF Blender dataset. We support the major datasets and allow users to create their own dataset, described in detail [here](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html).

```bash
ns-download-data --dataset=blender
ns-download-data --dataset=nerfstudio --capture=poster
```

Custom dataset: If you have custom data in the form of a video or folder of images, we've provided some [COLMAP](https://colmap.github.io/) and [FFmpeg](https://ffmpeg.org/download.html) scripts to help you process your data so it is compatible with nerfstudio.

After installing both software, you can process your data via:

```bash
ns-process-data {video,images,insta360} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
# Or if you're on a system without an attached display (i.e. colab):
ns-process-data {video,images,insta360} --data {DATA_PATH}  --output-dir {PROCESSED_DATA_DIR} --no-gpu
```

### Running NeRFStudio

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

## Training a model with the viewer

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

For more details on how to interact with the visualizer, please visit the official NeRFStudio viewer [walk-through](https://docs.nerf.studio/en/latest/quickstart/viewer_quickstart.html).

### Rendering a trajectory during inference

After your model has trained, you can headlessly render out a video of the scene with a pre-defined trajectory.

```bash
# assuming previously ran `ns-train nerfacto`
ns-render --load-config=outputs/data-nerfstudio-poster/nerfacto/{TIMESTAMP}/config.yml --traj=spiral --output-path=renders/output.mp4
```


# Phase 2: ORB-SLAM

## Setting up
We followed the directions from the [official ORB-SLAM3 github link](https://github.com/UZ-SLAMLab/ORB_SLAM3).

## Running ORB-SLAM3 with camera and custom data
We initially decided to use Monocular-Inertial data, but we had to switch to Stereo RGBD data as the ORB-SLAM3 library does not support outputting keyframe images using monocular data. We wrote our calibration file and we got to collect custom data, as shown. 

![](https://i.imgur.com/ZrkGNGT.gif)

We then received an output text file and converted the quaternion values into coordinates used by NeRFStudio with the `convert_txt_to_json.py` file.

<!-- # Phase 3: Testing -->
