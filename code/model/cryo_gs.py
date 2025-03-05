import os
import random
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pytorch_lightning as pl
import rich
import seaborn as sns
import tinycudann as tcnn
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import umap
from einops import rearrange, repeat
from scipy.cluster.vq import kmeans2

from ..utils import *
from .deformation import ImageEncoder
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.arguments import PipelineParams


class GaussianSplattingNeRF(nn.Module):
    def __init__(self, hetero_latent_dim=0, dtype=None, 
                 densify_from_iter=500, densify_until_iter=15000, 
                 densification_interval=100, args=None) -> None:
        super().__init__()
        
        self.dtype = dtype
        self.hetero_latent_dim = hetero_latent_dim
        
        # Adaptive control parameters
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        
        # Initialize Gaussian model
        self.gaussian_model = GaussianModel(scale_bound=None)
        initialize_gaussian(self.gaussian_model, args, None, random_init=True)

        # MLP for processing latent variables if heterogeneity is enabled
        if hetero_latent_dim > 0:
            self.latent_mlp = tcnn.Network(
                hetero_latent_dim, 
                hetero_latent_dim, 
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
                seed=random.randint(0, 1524367)
            )
    
    def forward(self, R, t, latent_variable=None):
        # Process latent variable if provided
        if latent_variable is not None and self.hetero_latent_dim > 0:
            latent_processed = self.latent_mlp(latent_variable)
        else:
            latent_processed = None
        
        # Use Gaussian Splatting for rendering
        img = self.render_with_gaussians(R, t, latent_processed)
        
        # Ensure output is float
        img = img.float()
        
        return img
    
    def render_with_gaussians(self, R, t, latent_variable=None):
        # Create a camera for rendering
        camera = self.create_camera_from_Rt(R, t, None)  # Pass None for image
        
        # Modify Gaussian parameters based on latent variable if needed
        if latent_variable is not None:
            # Apply deformation to Gaussian parameters based on latent variable
            self.apply_deformation(latent_variable)
        
        # Ensure all tensors in the Gaussian model are float32
        self.gaussian_model.to_float32()
        
        # Render using Gaussian Splatting
        dummy_parser = ArgumentParser()
        render_pkg = render(camera, self.gaussian_model, PipelineParams(dummy_parser))
        
        img = render_pkg["render"]
        
        return img
    
    def create_camera_from_Rt(self, R, t, image):
        """
        Create a camera for Gaussian Splatting based on the rotation and translation in CryoNeRF.
        
        Args:
            R: Rotation matrix (B, 3, 3)
            t: Translation vector (B, 3)
            
        Returns:
            camera: Camera object for Gaussian Splatting
        """
        from r2_gaussian.dataset.cameras import Camera
        t = - t.unsqueeze(1).bmm(R)
        # Convert R and t to numpy arrays for compatibility
        R_np = R.squeeze().cpu().numpy()
        t_np = t.squeeze().cpu().numpy()
    
        # Create a camera with parallel projection
        camera = Camera(
            colmap_id=0,
            scanner_cfg=None,
            R=R_np,
            T=t_np,
            angle=0.0,
            mode=0,
            FoVx=0.5,
            FoVy=0.5,
            image=image,
            image_name="",
            uid=0,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
        )
        
        # Set the mode to parallel beam (mode=0)
        camera.mode = 0
        
        return camera
    
    def apply_deformation(self, latent_variable):
        """
        Apply deformation to Gaussian parameters based on latent variable
        
        Args:
            latent_variable: Latent variable for heterogeneous reconstruction
        """
        # Get current Gaussian parameters
        means = self.gaussian_model.get_xyz
        scales = self.gaussian_model.get_scaling
        
        # Apply deformation based on latent variable
        # This is a simple implementation - you might want to use a more sophisticated approach
        deformation = latent_variable.mean(dim=0)
        deformation_scale = torch.sigmoid(deformation[:3]) * 0.1
        
        # Apply small deformation to positions and scales
        means_offset = means + deformation_scale
        scales_offset = scales * (1.0 + deformation_scale)
        
        # Update Gaussian parameters
        self.gaussian_model._xyz.data = means_offset
        self.gaussian_model._scaling.data = scales_offset

    def to_float32(self):
        """Convert all Gaussian model parameters to float32"""
        for param_name in ['_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest']:
            if hasattr(self.gaussian_model, param_name):
                param = getattr(self.gaussian_model, param_name)
                if param is not None and param.dtype != torch.float32:
                    setattr(self.gaussian_model, param_name, param.to(torch.float32))


class CryoGaussianSplatting(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Tell Lightning we're using manual optimization
        self.automatic_optimization = False
        
        # Add a custom step counter for manual optimization
        self._custom_step_counter = 0
        
        self.args = args
        self.size = args.size
        self.batch_size = args.batch_size
        self.ray_num = args.ray_num
        self.hetero_encoder_type = args.hetero_encoder_type
        self.hetero_latent_dim = args.hetero_latent_dim
        self.save_dir = args.save_dir
        self.log_vis_step = args.log_vis_step
        self.log_density_step = args.log_density_step
        self.print_step = args.print_step
        self.hetero = args.hetero
        
        # Adaptive control parameters
        self.densify_from_iter = getattr(args, 'densify_from_iter', 500)
        self.densify_until_iter = getattr(args, 'densify_until_iter', 15000)
        self.densification_interval = getattr(args, 'densification_interval', 100)
        self.densify_grad_threshold = getattr(args, 'densify_grad_threshold', 0.0005)
        self.density_min_threshold = getattr(args, 'density_min_threshold', 0.01)
        self.max_screen_size = getattr(args, 'max_screen_size', 20)
        self.max_scale = getattr(args, 'max_scale', 1.0)
        self.densify_scale_threshold = getattr(args, 'densify_scale_threshold', 0.01)
        self.max_num_gaussians = getattr(args, 'max_num_gaussians', 500000)

        if self.hetero:
            self.deformation_encoder = ImageEncoder(self.hetero_encoder_type, self.hetero_latent_dim if self.hetero else 0,
                                                  size=self.size, hartley=self.args.hartley)

        # Set up scanner configuration for volume querying
        self.scanner_cfg = {
            "offOrigin": torch.zeros(3),
            "nVoxel": torch.tensor([self.size, self.size, self.size]),
            "sVoxel": torch.tensor([1.0, 1.0, 1.0]),
            "dVoxel": torch.tensor([1.0/self.size, 1.0/self.size, 1.0/self.size])
        }
        self.ray_idx_all = repeat(torch.arange(
            self.size**2), "HW -> B HW D Dim3", B=self.batch_size, D=self.size, Dim3=3).long().cuda()

        x = np.linspace(0, 1.0, self.size, endpoint=False)
        y = np.linspace(0, 1.0, self.size, endpoint=False)
        z = np.linspace(0, 1.0, self.size, endpoint=False)
        self.volume_grid = torch.from_numpy(np.stack([coord.flatten()
                                            for coord in np.meshgrid(x, y, z, indexing='xy')], axis=1)).cuda().float()
        # Initialize Gaussian Splatting model instead of NeRF
        self.nerf = GaussianSplattingNeRF(
            hetero_latent_dim=self.hetero_latent_dim if self.hetero else 0,
            dtype=torch.float32,  # Force float32 instead of conditional precision
            densify_from_iter=self.densify_from_iter,
            densify_until_iter=self.densify_until_iter,
            densification_interval=self.densification_interval,
            args=self.scanner_cfg
        )

        # Define bounding box for Gaussians
        self.bbox = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

    def configure_optimizers(self):
        # Let the Gaussian model set up its own optimizer and learning rate schedulers
        from r2_gaussian.arguments import OptimizationParams
        from argparse import ArgumentParser
        
        # Create a dummy parser for OptimizationParams
        dummy_parser = ArgumentParser()
        
        # Create optimization parameters
        opt_params = OptimizationParams(dummy_parser)
        
        # Override with args if provided
        if hasattr(self.args, 'position_lr_init'):
            opt_params.position_lr_init = self.args.position_lr_init
        if hasattr(self.args, 'position_lr_final'):
            opt_params.position_lr_final = self.args.position_lr_final
        if hasattr(self.args, 'position_lr_max_steps'):
            opt_params.position_lr_max_steps = self.args.position_lr_max_steps
        
        if hasattr(self.args, 'density_lr_init'):
            opt_params.density_lr_init = self.args.density_lr_init
        if hasattr(self.args, 'density_lr_final'):
            opt_params.density_lr_final = self.args.density_lr_final
        if hasattr(self.args, 'density_lr_max_steps'):
            opt_params.density_lr_max_steps = self.args.density_lr_max_steps
        
        if hasattr(self.args, 'scaling_lr_init'):
            opt_params.scaling_lr_init = self.args.scaling_lr_init
        if hasattr(self.args, 'scaling_lr_final'):
            opt_params.scaling_lr_final = self.args.scaling_lr_final
        if hasattr(self.args, 'scaling_lr_max_steps'):
            opt_params.scaling_lr_max_steps = self.args.scaling_lr_max_steps
        
        if hasattr(self.args, 'rotation_lr_init'):
            opt_params.rotation_lr_init = self.args.rotation_lr_init
        if hasattr(self.args, 'rotation_lr_final'):
            opt_params.rotation_lr_final = self.args.rotation_lr_final
        if hasattr(self.args, 'rotation_lr_max_steps'):
            opt_params.rotation_lr_max_steps = self.args.rotation_lr_max_steps
        
        # Set up the Gaussian model's optimizer
        self.nerf.gaussian_model.training_setup(opt_params)
        
        # Create parameter groups for PyTorch Lightning
        param_groups = []
        
        # Add Gaussian model parameters - we'll use the optimizer directly in training_step
        # but we need to register it with Lightning
        param_groups.append({"params": [torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))], 
                             "lr": 0.0, "name": "dummy"})
        
        # Add other model parameters
        if self.hetero:
            param_groups.append(
                {"params": self.deformation_encoder.parameters(), "lr": 1e-4, "name": "encoder"})
            if hasattr(self.nerf, 'latent_mlp'):
                param_groups.append(
                    {"params": self.nerf.latent_mlp.parameters(), "lr": 1e-4, "name": "latent_mlp"})
        
        # Create a dummy optimizer for Lightning
        optimizer = torch.optim.AdamW(param_groups)
        
        # Create a scheduler for non-Gaussian parameters
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps, eta_min=1e-5)
        
        return {"optimizer": optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def render(self, R, t, latent_variable=None):
        pred_image = self.nerf.render_with_gaussians(R, t, latent_variable)
        return pred_image
    
    def render_density(self, latent_variable=None):
        """
        Query the density for the whole volume at once using Gaussian Splatting
        
        Args:
            latent_variable: Optional latent variable for heterogeneous reconstruction
            
        Returns:
            density: Density values for the whole volume (B, H, W, D)
        """
        # Apply deformation if latent variable is provided
        if latent_variable is not None and hasattr(self.nerf, 'apply_deformation'):
            self.nerf.apply_deformation(latent_variable)
        
        # Query the density using Gaussian Splatting
        center = self.scanner_cfg["offOrigin"]
        nVoxel = self.scanner_cfg["nVoxel"]
        sVoxel = self.scanner_cfg["sVoxel"]
        dummy_parser = ArgumentParser()
        # Query the whole volume at once
        query_result = query(
            self.nerf.gaussian_model,
            center,
            nVoxel,
            sVoxel,
            PipelineParams(dummy_parser)
        )
        
        # Get the volume from the query result
        density = query_result["vol"]
        
        # Reshape the density to (B, H, W, D)
        density = density.reshape(1, self.size, self.size, self.size)
        
        return density

    def on_train_start(self) -> None:
        # load all the image get the max and min
        if self.args.log_time:
            self.time_start = time.time()
            
        # Initialize the Gaussian model if not already initialized
        if not hasattr(self.nerf.gaussian_model, 'optimizer'):
            from r2_gaussian.arguments import OptimizationParams
            
            # Create a dummy parser for OptimizationParams
            dummy_parser = ArgumentParser()
            opt_params = OptimizationParams(dummy_parser)
            
            # Set optimization parameters from args if available
            if hasattr(self.args, 'lr_init'):
                opt_params.lr_init = self.args.lr_init
            if hasattr(self.args, 'lr_final'):
                opt_params.lr_final = self.args.lr_final
            if hasattr(self.args, 'lr_delay_steps'):
                opt_params.lr_delay_steps = self.args.lr_delay_steps
            if hasattr(self.args, 'lr_delay_mult'):
                opt_params.lr_delay_mult = self.args.lr_delay_mult
                
            # Setup training for the Gaussian model
            self.nerf.gaussian_model.training_setup(opt_params)
            
            # Initialize max_radii2D for adaptive control
            self.nerf.gaussian_model.max_radii2D = torch.zeros_like(self.nerf.gaussian_model.get_xyz[:, 0])

    def on_train_epoch_start(self) -> None:
        # Get dataset properties
        self.raw_size = self.trainer.train_dataloader.dataset.raw_size
        self.Apix = self.trainer.train_dataloader.dataset.Apix

    def training_step(self, batch, batch_idx):
        dummy_parser = ArgumentParser()
        # Get the optimizer
        opt = self.optimizers()
        # Use our custom step counter for optimization
        current_step = self._custom_step_counter
        
        R = batch["rotations"]
        t = batch["translations"] * 0.00000001
        N = R.shape[0]
        # Store current R and t for camera creation
        self.current_R = R
        self.current_t = t

        if self.hetero:
            latent_variable = self.deformation_encoder(batch["enc_images"].unsqueeze(1))
        else:
            latent_variable = None

        # Render the image using Gaussian Splatting
        camera = self.nerf.create_camera_from_Rt(R, t, batch["images"])
        
        # Apply deformation if latent variable is provided
        if latent_variable is not None and hasattr(self.nerf, 'apply_deformation'):
            self.nerf.apply_deformation(latent_variable)
        
        # Render using Gaussian Splatting
        render_pkg = render(camera, self.nerf.gaussian_model, PipelineParams(dummy_parser))
        
        pred_image = render_pkg["render"]
        
        # Get viewspace points and visibility for adaptive control
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        # print(self.nerf.gaussian_model.get_xyz.max(), self.nerf.gaussian_model.get_xyz.min())
        # vol_pred = query(
        #         self.nerf.gaussian_model,
        #         self.scanner_cfg["offOrigin"],
        #         self.scanner_cfg["nVoxel"],
        #         self.scanner_cfg["sVoxel"],
        #         PipelineParams(dummy_parser)
        #     )["vol"]
        # volume_grid_query = repeat(self.volume_grid, "HWD Dim3 -> B HWD Dim3", B=R.shape[0]).bmm(R) + (self.t_scale * t.unsqueeze(1)).bmm(R)
        # volume_grid_query = volume_grid_query.reshape(N, self.size**2, self.size, 3)
        # query the volume based on the volume_grid_query on the vol_pred
        # The volume_grid_query is the position of the gaussian in the volume
        # vol_pred is the density of the gaussian in the volume
        # the postion will out of the volume, so we need to clip it
        # volume_grid_query = torch.clamp(volume_grid_query, 0, self.size - 1)
        # position_in_volume = (volume_grid_query*self.scanner_cfg["nVoxel"]).long()
        # pred_image = vol_pred[position_in_volume].mean(-1)
        # Apply CTF corruption
        corrupted_pred_image = pred_image
        # torch.fft.fftshift(
        #     torch.fft.irfft2(
        #         torch.fft.rfft2(torch.fft.ifftshift(pred_image)) * torch.fft.fftshift(batch["ctfs"])[..., :self.size // 2 + 1]
        #     )
        # )

        # Calculate reconstruction loss
        loss_recon = F.mse_loss(corrupted_pred_image, batch["images"])
        loss = loss_recon
        
        # Manual backward for Gaussian model
        loss.backward()
        
        # Update Gaussian model parameters using our custom step counter
        self.nerf.gaussian_model.update_learning_rate(current_step)
        self.nerf.gaussian_model.optimizer.step()
        self.nerf.gaussian_model.optimizer.zero_grad(set_to_none=True)
        
        # Apply adaptive control for Gaussian Splatting
        with torch.no_grad():
            pass
            # Update maximum radii for each Gaussian
            if visibility_filter.any():
                self.nerf.gaussian_model.max_radii2D[visibility_filter] = torch.max(
                    self.nerf.gaussian_model.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                
                # Add densification statistics
                self.nerf.gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # Perform densification and pruning
                if (current_step > self.densify_from_iter and 
                    current_step < self.densify_until_iter and 
                    current_step % self.densification_interval == 0):
                    
                    # Calculate max_scale based on volume size
                    volume_to_world = max(self.scanner_cfg["sVoxel"])
                    max_scale = self.max_scale * volume_to_world if self.max_scale else None
                    densify_scale_threshold = (
                        self.densify_scale_threshold * volume_to_world
                        if self.densify_scale_threshold
                        else None
                    )
                    
                    # Perform densification and pruning
                    self.nerf.gaussian_model.densify_and_prune(
                        self.densify_grad_threshold,
                        self.density_min_threshold,
                        self.max_screen_size,
                        max_scale,
                        self.max_num_gaussians,
                        densify_scale_threshold,
                        self.bbox,
                    )
                    
                    # Log the number of Gaussians
                    if current_step % self.print_step == 0 and self.trainer.global_rank == 0:
                        rich.print(f"Number of Gaussians: {self.nerf.gaussian_model.get_xyz.shape[0]}")
        
        # Now handle the other parameters (encoder, latent_mlp) with Lightning's manual optimization
        if self.hetero:
            # Recompute the loss for the encoder and latent_mlp
            # We need to do this because we've already called backward() on the original loss
            
            # Zero the gradients for the encoder and latent_mlp
            opt.zero_grad()
            
            # Recompute the forward pass for the encoder
            latent_variable = self.deformation_encoder(batch["enc_images"].unsqueeze(1))
            
            if hasattr(self.nerf, 'latent_mlp'):
                latent_processed = self.nerf.latent_mlp(latent_variable)
            else:
                latent_processed = latent_variable
                
            # Apply deformation with the new latent
            self.nerf.apply_deformation(latent_processed)
            
            # Render again with the new parameters
            render_pkg = render(camera, self.nerf.gaussian_model, PipelineParams(dummy_parser))
            pred_image = render_pkg["render"]
            
            # Apply CTF corruption
            corrupted_pred_image = torch.fft.fftshift(
                torch.fft.irfft2(
                    torch.fft.rfft2(torch.fft.ifftshift(pred_image)) * torch.fft.fftshift(batch["ctfs"])[..., :self.size // 2 + 1]
                )
            )
            
            # Calculate reconstruction loss for the encoder and latent_mlp
            encoder_loss = F.mse_loss(corrupted_pred_image, batch["images"])
            
            # Optimize the encoder and latent_mlp
            self.manual_backward(encoder_loss)
            opt.step()
        
        # Query the volume for visualization
        if current_step % self.log_vis_step == 0 or current_step % self.log_density_step == 0:
            # Query the whole volume at once
            pred_density = self.render_density(
                latent_variable[0:1] if latent_variable is not None else None
            ).detach()

        if current_step % self.print_step == 0 and self.trainer.global_rank == 0:
            result = f"Current step: {current_step:06d}, " + \
                f"loss: {loss.item():.6f}, " + \
                f"loss_recon: {loss_recon.item():.6f}, " + \
                f"num_gaussians: {self.nerf.gaussian_model.get_xyz.shape[0]}"
            if latent_variable is not None:
                result += f", latent_norm: {torch.norm(latent_variable, p=2, dim=-1).mean().item():.6f}"
            rich.print(result)

        if current_step % self.log_vis_step == 0 and self.trainer.global_rank == 0:
            log_dir = f"{self.save_dir}/vis/{current_step:06d}"
            os.makedirs(log_dir, exist_ok=True)
            plt.imsave(f"{log_dir}/{current_step:06d}_pr.png", pred_image[0].detach().cpu().numpy(), cmap="gray")
            plt.imsave(f"{log_dir}/{current_step:06d}_cr.png", corrupted_pred_image[0].detach().cpu().numpy(), cmap="gray")
            plt.imsave(f"{log_dir}/{current_step:06d}_gt.png", batch["images"][0].cpu().numpy(), cmap="gray")
            plt.imsave(f"{log_dir}/{current_step:06d}_x.png",
                       pred_density[0, self.size // 2].cpu().numpy().transpose(), cmap="gray")
            plt.imsave(f"{log_dir}/{current_step:06d}_y.png",
                       pred_density[0, :, self.size // 2].cpu().numpy().transpose(), cmap="gray")
            plt.imsave(f"{log_dir}/{current_step:06d}_z.png",
                       pred_density[0, :, :, self.size // 2].cpu().numpy().transpose(), cmap="gray")

        if current_step % self.log_density_step == 0 and self.trainer.global_rank == 0:
            log_dir = f"{self.save_dir}/vis/{current_step:06d}"
            os.makedirs(log_dir, exist_ok=True)

            if not self.args.log_time:
                file_name = f"{log_dir}/{current_step:06d}_volume.mrc"
            else:
                time_elapsed = time.time() - self.time_start
                file_name = f"{log_dir}/{current_step:06d}_{time_elapsed:.4f}_volume.mrc"

            with mrcfile.new(file_name, overwrite=True) as mrc:
                density = pred_density[0].cpu().numpy().astype(np.float32)
                density = np.rot90(density, k=3, axes=(1, 2))
                density = np.rot90(density, k=2, axes=(0, 2))
                density = np.rot90(density, k=3, axes=(0, 1))
                mrc.set_data(density)
                mrc.set_volume()
                mrc.voxel_size = self.Apix
                mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0, 0, 0
                
            # Save Gaussian model checkpoint
            if hasattr(self.args, 'save_gaussian_model') and self.args.save_gaussian_model:
                ckpt_dir = f"{self.save_dir}/ckpt"
                os.makedirs(ckpt_dir, exist_ok=True)
                self.nerf.gaussian_model.save_ply(f"{ckpt_dir}/gaussians_{current_step:06d}.ply")

        # Log the loss for monitoring
        self.log("train_loss", loss.item(), prog_bar=True)
        
        # Increment our custom step counter
        self._custom_step_counter += 1
        
        return loss

    def on_validation_epoch_start(self):
        self.latent_vectors = []
        self.umap_model_2d = umap.UMAP()
        self.raw_size = self.trainer.val_dataloaders.dataset.raw_size
        self.Apix = self.trainer.val_dataloaders.dataset.Apix

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.hetero:
            self.latent_vectors.append(self.deformation_encoder(batch["enc_images"].unsqueeze(1)))
        else:
            return

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            if self.hetero:
                latent_variables = torch.cat(self.latent_vectors, dim=0)

                latent_2d = self.umap_model_2d.fit_transform(latent_variables.cpu().numpy())
                np.save(f"{self.save_dir}/latent_2d.npy", latent_2d)
                centroids, labels = kmeans2(latent_2d, k=6, seed=42)
                latent_2d_with_labels = np.column_stack((latent_2d, labels))

                sns.scatterplot(x=latent_2d_with_labels[:, 0], y=latent_2d_with_labels[:, 1],
                                hue=latent_2d_with_labels[:, 2], palette="viridis")
                plt.xlabel("UMAP1")
                plt.ylabel("UMAP2")
                plt.tight_layout()
                plt.savefig(f'{self.save_dir}/scatter_plot.png', dpi=300)
                plt.close()

                fig = sns.jointplot(x=latent_2d[:, 0], y=latent_2d[:, 1], kind="hex")
                fig.ax_joint.set_xlabel("UMAP1")
                fig.ax_joint.set_ylabel("UMAP2")
                plt.tight_layout()
                plt.savefig(f'{self.save_dir}/latent.png', dpi=300)
                plt.close()

                np.save(f"{self.save_dir}/latent_variables.npy", latent_variables.cpu().numpy())
            else:
                latent_variables = None

            if self.hetero:
                for i, label in enumerate(np.unique(labels)):
                    latent_variables_for_label = latent_variables[torch.from_numpy(labels == label)].mean(dim=0)

                    # Query the whole volume at once
                    pred_density = self.render_density(
                        latent_variables_for_label.unsqueeze(0).cuda()
                    ).detach().cpu()

                    with mrcfile.new(f"{self.save_dir}/volume_{i}.mrc", overwrite=True) as mrc:
                        density = pred_density[0].numpy().astype(np.float32)
                        density = np.rot90(density, k=3, axes=(1, 2))
                        density = np.rot90(density, k=2, axes=(0, 2))
                        density = np.rot90(density, k=3, axes=(0, 1))
                        mrc.set_data(density[::-1, :, :])
                        mrc.set_volume()
                        mrc.voxel_size = self.Apix
                        mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0, 0, 0
            else:
                # Query the whole volume at once
                pred_density = self.render_density().detach().cpu()

                with mrcfile.new(f"{self.save_dir}/volume.mrc", overwrite=True) as mrc:
                    density = pred_density[0].numpy().astype(np.float32)
                    density = np.rot90(density, k=3, axes=(1, 2))
                    density = np.rot90(density, k=2, axes=(0, 2))
                    density = np.rot90(density, k=3, axes=(0, 1))
                    mrc.set_data(density)
                    mrc.set_volume()
                    mrc.voxel_size = self.Apix
                    mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0, 0, 0
                    
            # Save final Gaussian model
            ckpt_dir = f"{self.save_dir}/ckpt"
            os.makedirs(ckpt_dir, exist_ok=True)
            self.nerf.gaussian_model.save_ply(f"{ckpt_dir}/gaussians_final.ply")

