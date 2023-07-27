import sys
import os
import time
import numpy as np
import torch
import argparse
import collections

sys.path.append(os.getcwd())
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from torchsummary import summary
from ptflops import get_model_complexity_info
from config.parse_config import ConfigParser
import src.dataset.build as module_data
import src.model.hnerv3d as module_arch
import src.evaluation.loss as module_loss
import src.evaluation.metric as module_metric
from src.trainer.nerv3d_trainer import NeRV3DTrainer


def main(config):
    logger = config.get_logger("train")

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    # Model
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    model = model.to(device)

    # Criterion & Metrics
    criterion = config.init_ftn("loss", module_loss)
    metrics = config.init_ftn("metrics", module_metric)

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
    args = argparse.ArgumentParser(description="NeRV3D Training")

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

    args.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        help="type of device",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(
            ["--fi", "--frame_interval"], type=int, target="dataset;args;frame_interval"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="dataloader;args;batch_size"
        ),
        CustomArgs(["--ep", "--epochs"], type=int, target="trainer;epochs"),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
