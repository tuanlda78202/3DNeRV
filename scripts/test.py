import argparse
import torch
import wandb
import numpy as np
from tqdm import tqdm
import src.dataset as module_data
import model.model as module_arch
from config.parse_config import ConfigParser
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

    # Config
    batch_size = config["data_loader"]["args"]["batch_size"]
    frame_interval = config["data_loader"]["args"]["frame_interval"]

    # Dataset & DataLoader
    build_data = config.init_ftn(
        "dataloader", module_data, batch_size=5, frame_interval=4
    )
    dataset, dataloader = build_data()

    # Model
    model = config.init_obj("arch", module_arch)

    # Global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    model = torch.compile(model.to(device))

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # Criterion & Metrics
    metrics = config.init_ftn("metrics", module_metric)

    model.eval()

    with torch.no_grad():
        psnr_video = []
        for batch_idx, data in enumerate(dataloader):
            data = data.permute(0, 4, 1, 2, 3).cuda()
            pred = model(data)

            # PSNR
            data = data.permute(0, 2, 1, 3, 4)
            psnr_batch = metrics(pred, data, bs=batch_size, fi=frame_interval)

            data = torch.mul(data, 255).type(torch.uint8)
            pred = torch.mul(pred, 255).type(torch.uint8)

            pred = pred.reshape(batch_size, frame_interval, 3, 720, 1280)
            data = data.reshape(batch_size, frame_interval, 3, 720, 1280)

            pred = pred.cpu().detach().numpy()
            data = data.cpu().detach().numpy()

            wandb.log(
                {
                    "psnr_batch": psnr_batch,
                    "pred": wandb.Video(pred, fps=4, format="mp4"),
                    "data": wandb.Video(data, fps=4, format="mp4"),
                },
            )

            psnr_video.append(psnr_batch)

            del pred, data

    wandb.log({"psnr_video": sum(psnr_video) / len(psnr_video)})

    wandb.finish()

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Inference with NeRV3D")
    args.add_argument(
        "-c",
        "--config",
        default="config/uvg-mp4-720p/beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="saved/models/Beauty-720pMP4-Flex3D-2.67M-300e/0720_004706/checkpoint-epoch299.pth",
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
