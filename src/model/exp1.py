import os, sys

sys.path.append(os.getcwd())
import torch
from src.backbone.videomae import vit_small_patch16_224
from collections import OrderedDict
from utils.util import load_yaml
from src.data.datasets import VideoDataSet


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


# args
file_path = "../checkpoint.pth"
model_key = "model|module"

# model
model = vit_small_patch16_224()

# Load checkpoint
checkpoint = torch.load(file_path, map_location="cpu")
print("Load ckpt from %s" % file_path)

for model_key in model_key.split("|"):
    if model_key in checkpoint:
        checkpoint_model = checkpoint[model_key]
        print("Load state_dict by model_key = %s" % model_key)
        break
if checkpoint_model is None:
    checkpoint_model = checkpoint
state_dict = model.state_dict()

for k in ["head.weight", "head.bias"]:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

all_keys = list(checkpoint_model.keys())
new_dict = OrderedDict()
for key in all_keys:
    if key.startswith("backbone."):
        new_dict[key[9:]] = checkpoint_model[key]
    elif key.startswith("encoder."):
        new_dict[key[8:]] = checkpoint_model[key]
    else:
        new_dict[key] = checkpoint_model[key]
checkpoint_model = new_dict

load_state_dict(model, checkpoint_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

config = load_yaml("/home/tuanlda78202/aaai24/configs/exp1/train_hnerv.yaml")
x = VideoDataSet(config)
