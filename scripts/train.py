import os
import sys
import torch
import argparse
import collections
import numpy as np

sys.path.append(os.getcwd())
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from config.parse_config import ConfigParser
import src.dataset.build as module_data
import src.model.nerv3d as module_arch
import src.evaluation.loss as module_loss
import src.evaluation.metric as module_metric
from src.trainer.nerv3d_trainer import NeRV3DTrainer


def main(config):
    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    model = config.init_obj("arch", module_arch)
    model = model.to(device)

    # Criterion & Metrics
    criterion = config.init_ftn("loss", module_loss)
    metrics = config.init_ftn("psnr", module_metric)

    # Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # Trainer
    trainer = NeRV3DTrainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        dataset=dataset,
        data_loader=dataloader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training NeRV3D")

    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(["-vp", "--valid_period"], type=int, target="trainer;valid_period")
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
