import sys
import os

sys.path.append(os.getcwd())

import argparse
import torch
import numpy as np
from tqdm import tqdm
from config.parse_config import ConfigParser
import src.dataset.build as module_data
from src.model.nerv3d import *
from src.compression.utils import *

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger("test")

    # Dataset
    build_data = config.init_ftn("dataloader", module_data)
    dataset, dataloader = build_data()

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

    total_time = []
    with torch.no_grad():
        for idx in enumerate(tqdm_batch):
            embed = torch.from_numpy(embedding[str(idx)]).cuda()
            pred, dec_time = decoder_model(embed)

            total_time.append(dec_time)

    print(
        "FPS: {}".format(
            len(dataset)
            * config["dataloader"]["args"]["frame_interval"]
            / sum(total_time)
        )
    )


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

    config = ConfigParser.from_args(args)

    main(config)
