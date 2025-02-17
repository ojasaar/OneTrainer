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
    
    # Check if we need to reshape based on actual dimensions
    if losses.shape[1] != mask.shape[1]:
        print(f"Dimension mismatch - adjusting mask channels")
        if losses.shape[1] == 16 and mask.shape[1] == 64:
            # Need to reduce mask channels from 64 to 16
            mask = mask.view(mask.shape[0], 16, 4, *mask.shape[2:]).mean(dim=2)
            print(f"Reduced mask shape: {mask.shape}")
        elif losses.shape[1] == 64 and mask.shape[1] == 16:
            # Need to expand mask channels from 16 to 64
            mask = mask.repeat_interleave(4, dim=1)
            print(f"Expanded mask shape: {mask.shape}")
        else:
            raise ValueError(f"Unsupported channel dimensions: losses {losses.shape[1]}, mask {mask.shape[1]}")

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
