import time
import torch
import deepCABAC
import numpy as np
from utils import load_yaml
from tqdm import tqdm


def dcabac_encoder(
    model,
    path=None,
    interv=0.1,
    stepsize=2 ** (-0.5 * 15),
    stepsize_other=2 ** (-0.5 * 19),
    _lambda=0.0,
):
    model_dict = model.state_dict().copy()
    ckpt_dict = load_yaml("src/compression/model.yaml")
    decoder_dict, embedding_dict = ckpt_dict["decoder"], ckpt_dict["embedding"]

    for key in embedding_dict:
        if key not in decoder_dict:
            del model_dict[key]

    encoder = deepCABAC.Encoder()

    t0 = time.time()

    for name, param in tqdm(model_dict.items()):
        param = param.cpu().numpy()
        if ".weight" in name:
            encoder.encodeWeightsRD(param, interv, stepsize, _lambda)
        else:
            encoder.encodeWeightsRD(param, interv, stepsize_other, _lambda)
    print(time.time() - t0)

    stream = encoder.finish().tobytes()
    print("Decoder compressed size: {} bits".format(8 * len(stream)))

    if path is not None:
        with open(path, "wb") as f:
            f.write(stream)


def dcabac_decoder(decoder_model, bin_path):
    with open(bin_path, "rb") as f:
        stream = f.read()

    decoder = deepCABAC.Decoder()
    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))

    decoder_dict = decoder_model.state_dict()

    for name in tqdm(decoder_dict.keys()):
        param = decoder.decodeWeights()
        decoder_dict[name] = torch.tensor(param)

    decoder.finish()

    decoder_model.load_state_dict(decoder_dict)

    return decoder_model.eval()
