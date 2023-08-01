import torch
import nnc
from utils import load_yaml
from src.model.nerv3d import *
from collections import defaultdict
from tqdm import tqdm


def state(full_model, raw_decoder_path):
    encoder_state, decoder_state = (
        full_model.state_dict().copy(),
        full_model.state_dict().copy(),
    )

    ckpt_dict = load_yaml("src/compression/model.yaml")
    decoder_list, encoder_list = ckpt_dict["decoder"], ckpt_dict["encoder"]

    for key in decoder_list:
        del encoder_state[key]  # Encoder (VMAE + Adaptive3D -> Embedding)

    for key in encoder_list:
        del decoder_state[key]  # Decoder

    torch.save(decoder_state, raw_decoder_path)

    encoder_model = NeRV3DEncoder(NeRV3D())
    encoder_model.load_state_dict(encoder_state)
    encoder_model.eval()

    return encoder_model


def dcabac_compress(raw_decoder_path, stream_path, qp, compressed_decoder_path):
    bit_stream = nnc.compress_model(
        raw_decoder_path, bitstream_path=stream_path, qp=qp, return_bitstream=True
    )
    nnc.decompress_model(stream_path, model_path=compressed_decoder_path)

    decoder_model = NeRV3DDecoder(NeRV3D())
    decoder_model.load_state_dict(torch.load(compressed_decoder_path))
    decoder_model.eval()

    return decoder_model, len(bit_stream)


def embedding_compress(dataloader, encoder_model, embedding_path, qp):
    embedding = defaultdict()

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Compress Embedding",
        total=len(dataloader),
        unit="it",
    )

    for batch_idx, data in enumerate(tqdm_batch):
        data = data.permute(0, 4, 1, 2, 3).cuda()
        feature = encoder_model(data)
        embedding[str(batch_idx)] = feature.cpu().detach().numpy()

    bit_stream = nnc.compress(
        parameter_dict=embedding,
        bitstream_path=embedding_path,
        qp=qp,
        return_bitstream=True,
    )

    embedding = nnc.decompress(embedding_path, return_model_information=True)

    return embedding[0], len(bit_stream)
