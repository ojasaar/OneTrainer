import torch
from torch import Tensor


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        is_flex_model: bool = False,
) -> Tensor:
    # Adjust mask channels if needed (64->16 for Flex model compatibility)
    if losses.shape[1] != mask.shape[1]:
        if losses.shape[1] == 16 and mask.shape[1] == 64:
            mask = mask.view(mask.shape[0], 16, 4, *mask.shape[2:]).mean(dim=2)
        else:
            raise ValueError(f"Unsupported channel dimensions: losses {losses.shape[1]}, mask {mask.shape[1]}")

    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    del clamped_mask
    return losses
