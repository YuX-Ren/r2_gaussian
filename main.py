import dataclasses
import os
from typing import Literal, Optional

import pytorch_lightning as pl
import rich
import tyro
from pytorch_lightning.callbacks import (ModelCheckpoint, RichProgressBar, TQDMProgressBar)
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from einops import repeat
from code.dataset import EMPIARDataset
from code.model import CryoGaussianSplatting


@dataclasses.dataclass
class Args:
    """Arguments of CryoNeRF."""
    
    dataset_dir: str
    """Root dir for datasets. It should be the parent folder of the dataset you want to reconstruct."""
    
    dataset: Literal["empiar-10028", "empiar-10076", "empiar-10049", "empiar-10180", "IgG-1D", "Ribosembly"]
    """Which dataset to use."""
    
    size: int = 256
    """Size of the volume and particle images."""

    batch_size: int = 1
    """Batch size for training."""
    
    ray_num: int = 8192
    """Number of rays to query in a batch."""
    
    nerf_hid_dim: int = 128
    """Hidden dim of NeRF."""
    
    nerf_hid_layer_num: int = 2
    """Number of hidden layers besides the input and output layer."""
    
    hetero_encoder_type: Literal["resnet18", "resnet34", "resnet50", "convnext_small", "convnext_base", ""] = "resnet18"
    """Encoder for deformation latent variable."""
    
    hetero_latent_dim: int = 16
    """Latent variable dim for deformation encoder."""
    
    save_dir: str = "experiments/test"
    """Dir to save visualization and checkpoint."""
    
    log_vis_step: int = 100
    """Number of steps to log visualization."""

    log_density_step: int = 10000
    """Number of steps to log density map."""
    
    print_step: int = 100
    """Number of steps to print once."""
    
    sign: Literal[1, -1] = -1
    """Sign of the particle images. For datasets used in the paper, this will be automatically set."""
    
    seed: int = -1
    """Whether to set a random seed. Default to not."""
    
    load_ckpt: Optional[str] = None
    """The checkpoint to load"""
    
    epochs: int = 1
    """Number of epochs for training."""
    
    hetero: bool = False
    """Whether to enable heterogeneous reconstruction."""
    
    val_only: bool = False
    """Only val"""
    
    first_half: bool = False
    """Whether to use the first half of the data to train for GSFSC computation."""
    
    second_half: bool = False
    """Whether to use the second half of the data to train for GSFSC computation."""
    # to float32 
    precision: str = "32"
    """The neumerical precision for all the computation. Recommended to set as default at 32."""

    max_steps: int = -1
    """The number of training steps. If set, this will supersede num_epochs."""

    log_time: bool = False
    """Whether to log the training time."""

    hartley: bool = True
    """Whether to encode the particle image in hartley space. This will improve heterogeneous reconstruction."""
    
class IterationProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if self.trainer.max_steps:
            bar.total = self.trainer.max_steps
        else:
            bar.total = self.trainer.num_training_batches
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        # Only reset if max_steps is not set
        if not self.trainer.max_steps:
            super().on_train_epoch_start(trainer, pl_module)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.total = self.trainer.num_val_batches[0] 
        return bar
    
    
class RichIterationProgressBar(RichProgressBar):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_disabled:
            return
        
        if trainer.max_steps > -1:
            total_batches = trainer.max_steps
        else:
            total_batches = self.total_train_batches
            
        train_description = "Training..."

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total_batches, train_description)
            else:
                self.progress.reset(
                    self.train_progress_bar_id,
                    total=total_batches,
                    description=train_description,
                    visible=True,
                )

        self.refresh()
        
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.dataset == "empiar-10028":
        sign = -1
    elif args.dataset == "empiar-10076":
        sign = 1
    elif args.dataset == "empiar-10049":
        sign = -1
    elif args.dataset == "empiar-10180":
        sign = -1
    elif args.dataset == "IgG-1D":
        sign = -1
    elif args.dataset == "Ribosembly":
        sign = -1
    else:
        sign = None
        rich.print("[red]Unknown dataset. Use sign specified in args![/red]")
    
    if args.load_ckpt:
        cryo_gs = CryoGaussianSplatting.load_from_checkpoint(args.load_ckpt, strict=True, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        cryo_gs = CryoGaussianSplatting(args=args)
        
    dataset = EMPIARDataset(
        mrcs=f"{args.dataset_dir}/{args.dataset}/particles.mrcs",
        ctf=f"{args.dataset_dir}/{args.dataset}/ctf.pkl",
        poses=f"{args.dataset_dir}/{args.dataset}/poses.pkl",
        args=args,
        size=args.size, sign=sign if sign is not None else args.sign,
    )
    # normalize the image to 0-1
    dataset.images = (dataset.images - dataset.images.min()) / (dataset.images.max() - dataset.images.min())
    # load the first image
    dataset.images = dataset.images[0:1]
    # contain the first image 10000 times
    dataset.images = repeat(dataset.images, 'b h w -> (b n) h w', n=10000)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False, pin_memory=True)
        
    logger = WandbLogger(name=f"CryoNeRF-{args.save_dir}", save_dir=args.save_dir, offline=True, project="CryoNeRF")
    logger.experiment.log_code(".")
    
    checkpoint_callback_step = ModelCheckpoint(dirpath=args.save_dir, save_top_k=-1, verbose=True, every_n_train_steps=20000, save_last=True)
    checkpoint_callback_epoch = ModelCheckpoint(dirpath=args.save_dir, save_top_k=-1, verbose=True, every_n_epochs=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="auto",
        max_epochs=args.epochs if args.max_steps == -1 else None,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=[RichIterationProgressBar(), checkpoint_callback_step, checkpoint_callback_epoch],
        precision=args.precision,
    )

    validator = pl.Trainer(
        accelerator="gpu",
        strategy=SingleDeviceStrategy(device="cuda:0"),
        max_epochs=args.epochs,
        logger=None,
        enable_checkpointing=False,
        enable_model_summary=False,
        devices=1,
        callbacks=[RichIterationProgressBar()],
        precision=args.precision,
    )
    
    if args.val_only:
        print(cryo_nerf)
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader, ckpt_path=args.load_ckpt)
    else:
        print(cryo_gs)
        trainer.fit(model=cryo_gs, train_dataloaders=train_dataloader, ckpt_path=args.load_ckpt)
        validator.validate(model=cryo_gs, dataloaders=valid_dataloader)