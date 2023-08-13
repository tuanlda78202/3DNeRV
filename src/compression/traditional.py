# Quantization + Huffman coding

from copy import deepcopy
from dahuffman import HuffmanCodec
import numpy as np
import torch


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
        "quantized_tensor": best_quantized_tensor,
        "min_value": quantization_params[best_index][0],
        "scale_factor": quantization_params[best_index][1],
    }

    return quantization_result, approximated_tensors[best_index]


def dequantize_tensor(quantization_data):
    quantized_tensor = quantization_data["quantized_tensor"]
    min_value = quantization_data["min_value"]
    scale_factor = quantization_data["scale_factor"]

    expanded_min_value = min_value.expand_as(quantized_tensor)
    expanded_scale_factor = scale_factor.expand_as(quantized_tensor)

    reconstructed_tensor = expanded_min_value + expanded_scale_factor * quantized_tensor
    return reconstructed_tensor


######################################################################################
def quantize_model(model, args):
    if args.quant_model_bit == -1:
        return model, None

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
                weight, args.quant_model_bit
            )
            quantized_state_dict[key] = quantization_data
            original_state_dict[key] = approximated_weight

    # Load the approximated weights back into the quantized model
    quantized_model.load_state_dict(original_state_dict)

    return quantized_model, quantized_state_dict


def huffman_encoding(quantized_embedding, quantized_weights):
    # Flatten the quantized embedding and extract the min and scale lengths
    quantized_values_list = quantized_embedding["quant"].flatten().tolist()
    min_scale_length = (
        quantized_embedding["min"].nelement() + quantized_embedding["scale"].nelement()
    )

    # Append the quantized values from the weights and update the min_scale_length
    for key, layer_weights in quantized_weights.items():
        quantized_values_list.extend(layer_weights["quant"].flatten().tolist())
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

    # Total bits per pixel (adapt as needed for your specific application)
    total_bits_per_pixel = total_bits

    print(
        f"After quantization and encoding: \n bits per parameter: {round(full_bits_per_param, 2)}, bits per pixel: {round(total_bits_per_pixel, 4)}"
    )

    return codec, symbol_bit_dictionary
