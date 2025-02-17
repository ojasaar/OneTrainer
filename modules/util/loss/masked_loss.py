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
    
    # Apply unmasked weight to non-masked regions
    clamped_mask = torch.where(mask > 0.5, torch.ones_like(mask), torch.ones_like(mask) * unmasked_weight)
    losses *= clamped_mask

    if normalize_masked_area_loss:
        # Calculate mean only over masked regions to avoid division by small numbers
        masked_mean = (losses * mask).sum(dim=(1, 2, 3), keepdim=True) / (mask.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)
        losses = losses / (masked_mean + 1e-6)

    del clamped_mask
    return losses
