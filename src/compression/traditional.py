# Quantization + Huffman coding

from copy import deepcopy
from dahuffman import HuffmanCodec
import numpy as np
import torch


def compute_quantization_params(t, bits=8):
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits - 1)
    tmin_scale_list = [[t_min, scale]]

    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits - 1)
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)])

    return tmin_scale_list


def quantize_tensor(t, t_min, scale, bits=8):
    t_min, scale = t_min.expand_as(t), scale.expand_as(t)
    quant_t = ((t - t_min) / scale).round().clamp(0, 2**bits - 1)
    new_t = t_min + scale * quant_t
    err_t = (t - new_t).abs().mean()
    return quant_t, new_t, err_t


def quant_tensor(t, bits=8):
    tmin_scale_list = compute_quantization_params(t, bits)
    quant_t_list, new_t_list, err_t_list = [], [], []

    for t_min, scale in tmin_scale_list:
        quant_t, new_t, err_t = quantize_tensor(t, t_min, scale, bits)
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)

    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    quant_t = {
        "quant": best_quant_t,
        "min": tmin_scale_list[best_quant_idx][0],
        "scale": tmin_scale_list[best_quant_idx][1],
    }

    return quant_t, new_t_list[best_quant_idx]


def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t["quant"], quant_t["min"], quant_t["scale"]
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t


######################################################################################
def quant_model(model, args):
    if args.quant_model_bit == -1:
        return model, None

    # Create a copy of the model and its state dictionary for quantization
    quant_model = deepcopy(model)
    quant_ckt, new_ckt = [quant_model.state_dict() for _ in range(2)]

    # Iterate through the state dictionary, quantizing weights (excluding encoders)
    for k, v in new_ckt.items():
        if "encoder" not in k:
            quant_v, new_v = quant_tensor(v, args.quant_model_bit)
            quant_ckt[k] = quant_v
            new_ckt[k] = new_v

    # Load the quantized weights back into the quantized model
    quant_model.load_state_dict(new_ckt)

    return quant_model, quant_ckt


def huffman_encoding(quant_embed, quant_ckt):
    quant_v_list = quant_embed["quant"].flatten().tolist()
    tmin_scale_len = quant_embed["min"].nelement() + quant_embed["scale"].nelement()
    for k, layer_wt in quant_ckt.items():
        quant_v_list.extend(layer_wt["quant"].flatten().tolist())
        tmin_scale_len += layer_wt["min"].nelement() + layer_wt["scale"].nelement()

    # Get the element name and its frequency
    unique, counts = np.unique(quant_v_list, return_counts=True)
    num_freq = dict(zip(unique, counts))

    # Generate HuffmanCoding table
    codec = HuffmanCodec.from_data(quant_v_list)
    sym_bit_dict = {}
    for k, v in codec.get_code_table().items():
        sym_bit_dict[k] = v[0]

    # Total bits for quantized embed + model weights
    total_bits = 0
    for num, freq in num_freq.items():
        total_bits += freq * sym_bit_dict[num]
    bits_per_param = total_bits / len(quant_v_list)

    # Including the overhead for min and scale storage
    total_bits += tmin_scale_len * 16  # (16 bits for float16)
    full_bits_per_param = total_bits / len(quant_v_list)

    # Bits per pixel (replace with appropriate calculation if needed)
    total_bpp = total_bits  # Adapt as needed

    print(
        f"After quantization and encoding: \n bits per parameter: {round(full_bits_per_param, 2)}, bits per pixel: {round(total_bpp, 4)}"
    )

    return codec, sym_bit_dict
