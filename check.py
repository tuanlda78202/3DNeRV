import torch

# Make sure we're using a NVIDIA GPU
if torch.cuda.is_available():
    # Get GPU capability score
    GPU_SCORE = torch.cuda.get_device_capability()
    print(f"GPU capability score: {GPU_SCORE}")
    if GPU_SCORE >= (8, 0):
        print(
            f"GPU score higher than or equal to (8, 0), PyTorch 2.x speedup features available."
        )
    else:
        print(
            f"GPU score lower than (8, 0), PyTorch 2.x speedup features will be limited (PyTorch 2.x speedups happen most on newer GPUs)."
        )

    # Print GPU info

else:
    print(
        "PyTorch couldn't find a GPU, to leverage the best of PyTorch 2.0, you should connect to a GPU."
    )
