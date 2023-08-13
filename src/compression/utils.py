import nnc
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from utils import load_yaml
from src.model.nerv3d import *
from dahuffman import HuffmanCodec


def state(full_model, raw_decoder_path, frame_interval):
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

    encoder_model = NeRV3DEncoder(NeRV3D(frame_interval=frame_interval))
    encoder_model.load_state_dict(encoder_state)
    encoder_model.eval()

    return encoder_model


# DeepCABAC
def embedding_compress(dataloader, encoder_model, embedding_path, eqp):
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
        qp=eqp,
        return_bitstream=True,
    )

    embedding = nnc.decompress(embedding_path, return_model_information=True)

    return embedding[0], len(bit_stream)


def dcabac_compress(
    raw_decoder_path, stream_path, mqp, compressed_decoder_path, decoder_dim
):
    bit_stream = nnc.compress_model(
        raw_decoder_path,
        bitstream_path=stream_path,
        qp=int(mqp),
        return_bitstream=True,
    )
    nnc.decompress_model(stream_path, model_path=compressed_decoder_path)

    decoder_model = NeRV3DDecoder(NeRV3D(decode_dim=decoder_dim))
    decoder_model.load_state_dict(torch.load(compressed_decoder_path))
    decoder_model.eval()

    return decoder_model, len(bit_stream)


def dcabac_decoding(embedding_path, stream_path, compressed_decoder_path, decoder_dim):
    embedding = nnc.decompress(embedding_path, return_model_information=True)

    nnc.decompress_model(stream_path, model_path=compressed_decoder_path)
    decoder_model = NeRV3DDecoder(NeRV3D(decode_dim=decoder_dim))
    decoder_model.load_state_dict(torch.load(compressed_decoder_path))
    decoder_model.eval()

    return embedding[0], decoder_model.cuda()


# Traditional Compression
def compute_quantization_params(tensor, num_bits=8):
    def calculate_scale(min_value, max_value):
        return (max_value - min_value) / (2**num_bits - 1)

    global_min, global_max = tensor.min(), tensor.max()
    quantization_params = [[global_min, calculate_scale(global_min, global_max)]]

    for axis in range(tensor.dim()):
        axis_min, axis_max = (
            tensor.min(axis, keepdim=True)[0],
            tensor.max(axis, keepdim=True)[0],
        )

        if axis_min.nelement() / tensor.nelement() < 0.02:
            axis_scale = calculate_scale(axis_min, axis_max).to(torch.float16)
            quantization_params.append([axis_min.to(torch.float16), axis_scale])

    return quantization_params


def quantize_tensor(tensor, min_value, scale_factor, num_bits=8):
    expanded_min_value = min_value.expand_as(tensor)
    expanded_scale_factor = scale_factor.expand_as(tensor)

    quantized_tensor = (
        ((tensor - expanded_min_value) / expanded_scale_factor)
        .round()
        .clamp(0, 2**num_bits - 1)
    )
    approximated_tensor = expanded_min_value + expanded_scale_factor * quantized_tensor
    quantization_error = (tensor - approximated_tensor).abs().mean()

    return quantized_tensor, approximated_tensor, quantization_error


def quantize_tensor_full(tensor, num_bits=8):
    quantization_params = compute_quantization_params(tensor, num_bits)
    quantized_tensors, approximated_tensors, errors = [], [], []

    for min_value, scale_factor in quantization_params:
        quantized_tensor, approximated_tensor, error = quantize_tensor(
            tensor, min_value, scale_factor, num_bits
        )
        quantized_tensors.append(quantized_tensor)
        approximated_tensors.append(approximated_tensor)
        errors.append(error)

    best_error = min(errors)
    best_index = errors.index(best_error)
    best_quantized_tensor = quantized_tensors[best_index].to(torch.uint8)

    quantization_result = {
        "quantized": best_quantized_tensor,
        "min": quantization_params[best_index][0],
        "scale": quantization_params[best_index][1],
    }

    return quantization_result, approximated_tensors[best_index]


def quantize_model(model, num_bits=8):
    # Create a copy of the model for quantization
    quantized_model = deepcopy(model)

    # Retrieve the state dictionary for both quantized and original weights
    quantized_state_dict, original_state_dict = [
        quantized_model.state_dict() for _ in range(2)
    ]

    # Iterate through the original weights, quantizing all except encoders
    for key, weight in original_state_dict.items():
        if "encoder" not in key:
            quantization_data, approximated_weight = quantize_tensor_full(
                weight, num_bits
            )

            quantized_state_dict[key] = quantization_data
            original_state_dict[key] = approximated_weight

    # Load the approximated weights back into the quantized model
    quantized_model.load_state_dict(original_state_dict)

    return quantized_model, quantized_state_dict


def huffman_encoding(quantized_embedding, quantized_weights):
    # Embedding
    quantized_values_list = quantized_embedding["quantized"].flatten().tolist()
    min_scale_length = (
        quantized_embedding["min"].nelement() + quantized_embedding["scale"].nelement()
    )

    # Decoder weights
    for key, layer_weights in quantized_weights.items():
        quantized_values_list.extend(layer_weights["quantized"].flatten().tolist())

        min_scale_length += (
            layer_weights["min"].nelement() + layer_weights["scale"].nelement()
        )

    # Compute the frequency of each unique value
    unique_values, counts = np.unique(quantized_values_list, return_counts=True)
    value_frequency = dict(zip(unique_values, counts))

    # Generate Huffman coding table
    codec = HuffmanCodec.from_data(quantized_values_list)
    symbol_bit_dictionary = {
        key: value[0] for key, value in codec.get_code_table().items()
    }

    # Compute total bits for quantized data
    total_bits = sum(
        freq * symbol_bit_dictionary[num] for num, freq in value_frequency.items()
    )

    # Include the overhead for min and scale storage
    total_bits += min_scale_length * 16  # (16 bits for float16)
    full_bits_per_param = total_bits / len(quantized_values_list)

    return full_bits_per_param, total_bits


def dequantize_tensor(quantization_data):
    quantized_tensor = quantization_data["quantized"]
    min_value = quantization_data["min"]
    scale_factor = quantization_data["scale"]

    expanded_min_value = min_value.expand_as(quantized_tensor)
    expanded_scale_factor = scale_factor.expand_as(quantized_tensor)

    reconstructed_tensor = expanded_min_value + expanded_scale_factor * quantized_tensor
    return reconstructed_tensor


def normal_compression(
    dataloader,
    encoder_model,
    raw_decoder_path,
    traditional_embedding_path,
    decoder_dim,
    frame_interval,
):
    # Embedding
    embedding_list = []

    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Processing Embedding",
        total=len(dataloader),
        unit="it",
    )

    for batch_idx, data in enumerate(tqdm_batch):
        data = data.permute(0, 4, 1, 2, 3).cuda()
        feature = encoder_model(data)
        embedding_list.append(feature.cpu().detach().numpy())

    torch.save(torch.Tensor(embedding_list), traditional_embedding_path)

    # Decoder
    decoder_model = NeRV3DDecoder(
        NeRV3D(frame_interval=frame_interval, decode_dim=decoder_dim)
    )
    decoder_model.load_state_dict(torch.load(raw_decoder_path))
    decoder_model.eval()

    # Quantization
    quantize_embedding, _ = quantize_tensor_full(
        torch.load(traditional_embedding_path), num_bits=6
    )
    quantized_model, quantized_model_state = quantize_model(decoder_model, num_bits=8)

    # Huffman Encoding
    bits_per_param, total_bits = huffman_encoding(
        quantize_embedding, quantized_model_state
    )

    # Dequantization
    embedding = dequantize_tensor(quantize_embedding)

    return embedding, quantized_model, total_bits
