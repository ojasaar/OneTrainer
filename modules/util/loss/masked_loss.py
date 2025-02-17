import torch
from torch import Tensor


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        is_flex_model: bool = False,
) -> Tensor:
    # If mask has fewer channels than losses, repeat it to match
    if losses.shape[1] != mask.shape[1]:
        if mask.shape[1] == 1:
            mask = mask.repeat(1, losses.shape[1], 1, 1)
        else:
            raise ValueError(f"Unsupported channel dimensions: losses {losses.shape[1]}, mask {mask.shape[1]}")

    # Ensure mask is binary
    mask = (mask > 0.5).to(mask.dtype)
    
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    del clamped_mask
    return losses
