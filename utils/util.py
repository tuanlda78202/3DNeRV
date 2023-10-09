import os
import yaml
from pathlib import Path


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


def make_dir(path):
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
