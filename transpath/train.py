import argparse
import os

import pytorch_lightning as pl
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path
from data.hmaps import GridData
from models.autoencoder import Autoencoder, PathLogger

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(mode, run_name, proj_name, batch_size, max_epochs, img_size, ckpt_path=None):
    train_data = GridData(
        path='./TransPath_data/train',
        mode=mode,
        img_size=img_size
    )
    val_data = GridData(
        path='./TransPath_data/val',
        mode=mode,
        img_size=img_size
    )
    resolution = (train_data.img_size, train_data.img_size)
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True
                                  )
    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True
                                )

    samples = next(iter(val_dataloader))

    model = Autoencoder(mode=mode, resolution=resolution)

    if ckpt_path:
        print('Loading Checkpoint')

        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path)
        ckpt_resolution = ckpt['hyper_parameters']['resolution']
        ckpt_model = Autoencoder(mode=mode, resolution=ckpt_resolution)
        ckpt_model.load_state_dict(ckpt['state_dict'])

        model.encoder.load_state_dict(ckpt_model.encoder.state_dict())
        model.decoder.load_state_dict(ckpt_model.decoder.state_dict())
        model.transformer.load_state_dict(ckpt_model.transformer.state_dict())

    callback = PathLogger(samples, mode=mode)

    # Get the W&B API key from the environment variable
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    wandb_logger = WandbLogger(project=proj_name, name=f'{run_name}_{mode}', log_model='all')
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        max_epochs=max_epochs,
        deterministic=False,
        callbacks=[callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='f')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--proj_name', type=str, default='TransPath_runs')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=160)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--ckpt_path', type=str, required=False)

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high') #fix for tesor blocks warning with new video card
    main(
        mode=args.mode,
        run_name=args.run_name,
        proj_name=args.proj_name,
        batch_size=args.batch,
        max_epochs=args.epoch,
        img_size=args.img_size,
        ckpt_path=args.ckpt_path
    )
