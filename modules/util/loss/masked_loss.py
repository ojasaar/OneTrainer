import torch
from torch import Tensor


def masked_losses(
        losses: Tensor,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        is_flex_model: bool = False,
) -> Tensor:
    print(f"\nMasked Losses Debug:")
    print(f"Input losses shape: {losses.shape}")
    print(f"Input mask shape: {mask.shape}")
    print(f"Is Flex model: {is_flex_model}")

    # For Flex models, reshape the mask from [16, h, w] to [64, h, w] to match the model's output dimensions
    if is_flex_model:
        # Repeat each channel 4 times to go from 16 to 64 channels
        mask = mask.repeat_interleave(4, dim=0)
        print(f"Reshaped mask shape (after repeat_interleave): {mask.shape}")

    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    print(f"Clamped mask shape: {clamped_mask.shape}")
    print(f"Before multiplication - losses shape: {losses.shape}, clamped_mask shape: {clamped_mask.shape}")
    
    losses *= clamped_mask
    print(f"After multiplication - losses shape: {losses.shape}")

    if normalize_masked_area_loss:
        mean_shape = clamped_mask.mean(dim=(1, 2, 3), keepdim=True).shape
        print(f"Mean mask shape for normalization: {mean_shape}")
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)
        print(f"After normalization - losses shape: {losses.shape}")

    del clamped_mask

    return losses
