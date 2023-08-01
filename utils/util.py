########## HNeRV
import torch
import torch.nn as nn
from math import ceil
from pathlib import Path
import yaml
import os


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def load_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as file:
        config = yaml.safe_load(file)
    return config


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        yaml.dump(content, handle, indent=4, sort_keys=False)


def init_wandb(
    wandb_lib,
    project,
    entity,
    api_key_file,
    mode="online",
    dir="./saved",
    name=None,
    config=None,
):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert os.path.exists(api_key_file), "The given W&B API key file does not exist"

    # Set environment API & DIR
    api_key_value = open(api_key_file, "r").read().strip()
    os.environ["WANDB_API_KEY"] = api_key_value
    os.environ["WANDB_DIR"] = dir
    os.environ["WANDB_MODE"] = mode

    # name: user_name in WandB
    return wandb_lib.init(project=project, entity=entity, name=name, config=config)


def state(full_model, raw_decoder_path):
    # State Dict
    encoder_state, decoder_state = (
        full_model.state_dict().copy(),
        full_model.state_dict().copy(),
    )

    ckpt_dict = load_yaml("config/model.yaml")
    decoder_list, encoder_list = ckpt_dict["decoder"], ckpt_dict["encoder"]

    for key in decoder_list:
        del encoder_state[key]  # Encoder (VMAE + Adaptive3D -> Embedding)

    for key in encoder_list:
        del decoder_state[key]  # Decoder

    torch.save(decoder_state, raw_decoder_path)

    return encoder_state


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader
