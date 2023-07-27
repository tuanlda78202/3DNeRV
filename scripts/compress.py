import sys
import os

sys.path.append(os.getcwd())
os.environ["WANDB_DIR"] = "./saved"

import argparse
import torch
import wandb
import numpy as np
from tqdm import tqdm
import src.model as module_arch
from config.parse_config import ConfigParser
from src.compression.dcabac import *
import src.dataset.build as module_data
import src.model.hnerv3d as module_arch
import src.evaluation.metric as module_metric


np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger("test")

    # Dataset & DataLoader
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

    del dataset

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    model = config.init_obj("arch", module_arch).to(device)

    # Criterion & Metrics
    metrics = config.init_ftn("metrics", module_metric)

    # CKPT
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Encoder
    # dcabac_encoder(model)

    # Decoder


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Inference with NeRV3D")
    args.add_argument(
        "-c",
        "--config",
        default="src/compression/test.yaml",  # check.yaml
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default="../ckpt/720p/bee-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_ckpte300.pth",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)

    main(config)
