"""
Implementation of NeSLAM.

Requirement:
- Need to install torch per:
https://docs.nerf.studio/en/latest/quickstart/installation.html#dependencies

Creating NeSLAM model based off of nerfstudio documentation:
    1. https://docs.nerf.studio/en/latest/developer_guides/pipelines/models.html
    2. https://docs.nerf.studio/en/latest/developer_guides/config.html
and the 3 detailed-focus layer MLP seen in NICE-SLAM:
https://github.com/cvg/nice-slam/blob/master/src/conv_onet/models/decoder.py

NeSLAM, NICE-SLAM, and Mip-Nerf probably work better because of Gaussian Fourier Feature Transforms
and that all of them use multiple MLPs (or voxel grids in NICE-SLAM's case) to detect and use at least
2 detail layers: fine and coarse. NeSLAM and Mip-Nerf take it up a notch by also detecting and using
a medium detail layer.
https://github.com/cvg/nice-slam/blob/master/src/conv_onet/models/decoder.py
and https://github.com/ciglenecki/nerf-research (mip-nerf)
Mip-Nerf paper: https://arxiv.org/pdf/2103.13415.pdf

Need overall view?
https://github.com/nerfstudio-project/nerfstudio/blob/afd6fec6e73557e662978af742fd1c464e92f556/docs/nerfology/methods/nerfacto.md

If you want to delve deeper into MLP setup by going through what goes before the nerfacto and instant-ngp models
https://github.com/nerfstudio-project/nerfstudio/blob/b8f85fb603e426309697f7590db3e2c34b9a0d66/nerfstudio/fields/nerfacto_field.py
https://github.com/nerfstudio-project/nerfstudio/blob/b8f85fb603e426309697f7590db3e2c34b9a0d66/nerfstudio/fields/instant_ngp_field.py

"""

from __future__ import annotations

#from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type
#from typing_extensions import Literal

#import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle

from nerfstudio.models.base_model import Model, ModelConfig
# from nerfstudio.models.nerfacto import NerfactoModelConfig

# from nerfstudio.field_components.spatial_distortions import SceneContraction
# from nerfstudio.fields.neslam_field import TCNNNeSLAMField
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.losses import MSELoss#, distortion_loss, interlevel_loss

# Had class NeSLAMModelConfig(NerfactoModelConfig) before inspired from models/nerfacto.py

class NeSLAMModel(Model):
    """NeSLAM model

    Args:
        config: NeSLAM configuration to instantiate model
    """

    # config: NeSLAMModelConfig
    def __init__(
        self,
        config: ModelConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_medium = None
        self.field_fine = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        """
        # Using a Scene Contraction will bind the space to a cube of
        # side length 4, which will be useful if downstream encoders
        # operate on the grid (eg. hashencoder in Instant-NGP)
        # https://docs.nerf.studio/en/latest/nerfology/model_components/visualize_spatial_distortions.html
        scene_contraction = SceneContraction(order=float("inf"))
    
        # Fields - will use https://github.com/NVlabs/tiny-cuda-nn, like other models do
        self.field = TCNNNeSLAMField(
            self.scene_box.aabb,
            num_layers=4,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )
        """
        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        ) #num_freq=16, max_freq_exp=16
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # From mip-nerf
        # self.field = NeRFField(
        #     position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True
        # )

        # From vanilla_nerf
        self.field_coarse = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_medium = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)
        self.field_fine = NeRFField(position_encoding=position_encoding, direction_encoding=direction_encoding)

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)
        #, include_original=False) # for mip-nerf

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimizer your model components."""
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        # param_groups["fields"] = list(self.field.parameters()) # From mipnerf
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_medium.parameters()) + list(self.field_fine.parameters())
        return param_groups

    # def get_training_callbacks(
    #         self, training_callback_attributes: TrainingCallbackAttributes
    #     ) -> List[TrainingCallback]:
    #     """Returns the training callbacks, such as updating a density grid for Instant NGP."""

    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""

        if (self.field_coarse is None) or (self.field_medium is None) or (self.field_fine is None):
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass:
        field_outputs_medium = self.field_medium.forward(ray_samples_pdf)
        weights_medium = ray_samples_pdf.get_weights(field_outputs_medium[FieldHeadNames.DENSITY])
        rgb_medium = self.renderer_rgb(
            rgb=field_outputs_medium[FieldHeadNames.RGB],
            weights=weights_medium,
        )
        accumulation_medium = self.renderer_accumulation(weights_medium)
        depth_medium = self.renderer_depth(weights_medium, ray_samples_pdf)

        # pdf sampling again
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_medium)

        # Third pass:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_medium": rgb_medium,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_medium": accumulation_medium,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_medium": depth_medium,
            "depth_fine": depth_fine,
        }
        return outputs

    # def get_metrics_dict(self, outputs, batch):
    #     """Returns metrics dictionary which will be plotted with wandb or tensorboard."""

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)

        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_medium = self.rgb_loss(image, outputs["rgb_medium"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_medium": rgb_loss_medium, "rgb_loss_fine": rgb_loss_fine}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_medium = outputs["rgb_medium"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_medium = colormaps.apply_colormap(outputs["accumulation_medium"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_medium = colormaps.apply_depth_colormap(
            outputs["depth_medium"],
            accumulation=outputs["accumulation_medium"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_medium, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_medium, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_medium, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_medium = torch.moveaxis(rgb_medium, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        # Done in mipnerf but not in vanilla_nerf
        # rgb_coarse = torch.clip(rgb_coarse, min=-1, max=1)
        # rgb_medium = torch.clip(rgb_medium, min=-1, max=1)
        # rgb_fine = torch.clip(rgb_fine, min=-1, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        medium_psnr = self.psnr(image, rgb_medium)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "medium-psnr": float(medium_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        return metrics_dict, images_dict
