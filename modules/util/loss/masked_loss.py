import torch
from torch import Tensor


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        is_flex_model: bool = False,
) -> Tensor:
    # For Flex models, reshape the mask from [16, h, w] to [64, h, w] to match the model's output dimensions
    if is_flex_model:
        # Reshape [16, h, w] to [4, 4, h, w]
        mask = mask.view(4, 4, *mask.shape[1:])
        # Permute to [4, 4, h, w]
        mask = mask.permute(0, 1, 2, 3)
        # Reshape to [16, h, w]
        mask = mask.reshape(64, *mask.shape[2:])

    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    del clamped_mask

    return losses
