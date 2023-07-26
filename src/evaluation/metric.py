from typing import Tuple, List, Optional, Union, Dict, Any
import torch


def psnr_one(pred, gt):
    EPS = 1e-8
    mse = torch.mean((pred - gt) ** 2)
    db = -10 * torch.log10(mse + EPS)  # bcs normalized
    return db.item()


def psnr_batch(batch_pred, batch_gt, batch_size=None, frame_interval=None):
    psnr_list = []

    for batch_idx in range(batch_size):
        for fi_idx in range(frame_interval):
            psnr_list.append(
                psnr_one(batch_pred[batch_idx][fi_idx], batch_gt[batch_idx][fi_idx])
            )

    result = sum(psnr_list) / len(psnr_list)
    return round(result, 2)


def _validate_input(
    tensors: List[torch.Tensor],
    dim_range: Tuple[int, int] = (0, -1),
    data_range: Tuple[float, float] = (0.0, -1.0),
    size_range: Optional[Tuple[int, int]] = None,
    check_for_channels_first: bool = False,
) -> None:
    r"""Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f"Expected torch.Tensor, got {type(t)}"
        assert (
            t.device == x.device
        ), f"Expected tensors to be on {x.device}, got {t.device}"

        if size_range is None:
            assert (
                t.size() == x.size()
            ), f"Expected tensors with same size, got {t.size()} and {x.size()}"
        else:
            assert (
                t.size()[size_range[0] : size_range[1]]
                == x.size()[size_range[0] : size_range[1]]
            ), f"Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}"

        if dim_range[0] == dim_range[1]:
            assert (
                t.dim() == dim_range[0]
            ), f"Expected number of dimensions to be {dim_range[0]}, got {t.dim()}"
        elif dim_range[0] < dim_range[1]:
            assert (
                dim_range[0] <= t.dim() <= dim_range[1]
            ), f"Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}"

        if data_range[0] < data_range[1]:
            assert (
                data_range[0] <= t.min()
            ), f"Expected values to be greater or equal to {data_range[0]}, got {t.min()}"
            assert (
                t.max() <= data_range[1]
            ), f"Expected values to be lower or equal to {data_range[1]}, got {t.max()}"

        if check_for_channels_first:
            channels_last = t.shape[-1] in {1, 2, 3}
            assert (
                not channels_last
            ), "Expected tensor to have channels first format, but got channels last. \
                Please permute channels (e.g. t.permute(0, 3, 1, 2) for 4D tensors) and rerun."


def _reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.

    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == "none":
        return x
    elif reduction == "mean":
        return x.mean(dim=0)
    elif reduction == "sum":
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: Union[int, float] = 1.0,
    reduction: str = "mean",
    convert_to_greyscale: bool = False,
) -> torch.Tensor:
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        convert_to_greyscale: Convert RGB image to YIQ format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.

    Returns:
        PSNR Index of similarity between two images.

    References:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    # Constant for numerical stability
    EPS = 1e-8

    x = x / float(data_range)
    y = y / float(data_range)

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YIQ and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor(
            [0.299, 0.587, 0.114], device=x.device, dtype=x.dtype
        ).view(1, -1, 1, 1)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score: torch.Tensor = -10 * torch.log10(mse + EPS)

    return _reduce(score, reduction)
