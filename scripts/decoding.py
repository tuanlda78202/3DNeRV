import sys
import os

sys.path.append(os.getcwd())

import argparse
import torch
import numpy as np
from tqdm import tqdm
from config.parse_config import ConfigParser
from src.model.nerv3d import *
from src.compression.utils import *

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger("test")

    # Embedding & Decoder
    embedding, decoder_model = dcabac_decoding(
        embedding_path=config["compression"]["embedding_path"],
        stream_path=config["compression"]["stream_path"],
        compressed_decoder_path=config["compression"]["compressed_decoder_path"],
        decoder_dim=config["arch"]["args"]["decode_dim"],
    )

    tqdm_batch = tqdm(
        iterable=range(args.frames),
        desc="Decoding UVG",
        total=args.frames,
        unit="it",
    )

    with torch.no_grad():
        for idx in enumerate(tqdm_batch):
            embed = torch.from_numpy(embedding[str(idx)]).cuda()
            pred = decoder_model(embed)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="NeRV3D Compression")

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
        "-f",
        "--frames",
        default=600,
        type=int,
        help="Number of frames",
    )

    config = ConfigParser.from_args(args)

    main(config)
