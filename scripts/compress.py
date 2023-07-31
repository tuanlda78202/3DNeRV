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
from src.model.hnerv3d import *
from utils import load_yaml
from torchsummary import summary

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

    # DeepCABAC
    model_dict = model.state_dict().copy()
    ckpt_dict = load_yaml("src/compression/model.yaml")
    decoder_dict, embedding_dict = ckpt_dict["decoder"], ckpt_dict["embedding"]

    for key in embedding_dict:
        if key not in decoder_dict:
            del model_dict[key]

    decoder_model = HNeRVMaeDecoder(HNeRVMae())
    # print(summary(decoder_model, (144, 16)))

    dcabac(model, decoder_model)

    feature_encoder = HNeRVMaeEncoder(model).cuda()
    # DeepCABAC Decoder

    for batch_idx, data in enumerate(dataloader):
        data = data.permute(0, 4, 1, 2, 3).cuda()

        feature = feature_encoder(data)

        print(feature.shape)

    """"
    decoder_model = dcabac_decoder(
        HNeRVMaeDecoder(HNeRVMae()), bin_path="src/compression/beauty.bin"
    )
    print(model)
    """
    """
    for batch_idx, data in enumerate(dataloader):
        data = data.permute(0, 4, 1, 2, 3).cuda()

        features = pretrained_mae.forward_features(data)
        output = decoder_model(features)

        # PSNR
        pred = output
        gt = data.permute(0, 2, 1, 3, 4)

        loss = F.mse_loss(pred, gt)
        psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=FRAME_INTERVAL)
        print(loss, psnr_db)

        del pred, gt, output, data
    """


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
