from abc import ABCMeta

import torch
from torch import Tensor
from modules.util.config.TrainConfig import TrainConfig


class ModelSetupFlowMatchingMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.__sigma = None
        self.__one_minus_sigma = None

    def _add_noise_discrete(
            self,
            scaled_latent_image: Tensor,
            latent_noise: Tensor,
            timestep: Tensor,
            timesteps: Tensor,
            *,
            batch: dict = None,
            config: TrainConfig = None,
    ) -> tuple[Tensor, Tensor]:
        # Debug logging to identify context
        is_training = batch is not None and 'latent_mask' in batch
        print(f"[FlowMatchingMixin] Context: {'Training' if is_training else 'Sampling'}")
        if config:
            print(f"[FlowMatchingMixin] noise_mask enabled: {config.noise_mask}")

        if self.__sigma is None:
            num_timesteps = timesteps.shape[-1]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=scaled_latent_image.device)
            self.__sigma = all_timesteps / num_timesteps
            self.__one_minus_sigma = 1.0 - self.__sigma

        orig_dtype = scaled_latent_image.dtype

        sigmas = self.__sigma[timestep]
        one_minus_sigmas = self.__one_minus_sigma[timestep]

        while sigmas.dim() < scaled_latent_image.dim():
            sigmas = sigmas.unsqueeze(-1)
            one_minus_sigmas = one_minus_sigmas.unsqueeze(-1)

        # Calculate the noisy latent image
        scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas \
                                    + scaled_latent_image.to(dtype=sigmas.dtype) * one_minus_sigmas

        # If noise masking is enabled, only apply noise in the masked region
        if config and config.noise_mask and batch and 'latent_mask' in batch:
            # Binarize the mask
            binary_mask = (batch['latent_mask'] > 0.5).to(batch['latent_mask'].dtype)
            print(f"[FlowMatchingMixin] mask shape: {binary_mask.shape}, unique values after binarization: {torch.unique(binary_mask).tolist()}")
            
            print("[FlowMatchingMixin] Applying noise masking")
            print(f"[FlowMatchingMixin] Before masking - noisy image range: {scaled_noisy_latent_image.min():.3f} to {scaled_noisy_latent_image.max():.3f}")
            
            # Apply noise in masked regions (where mask is 1), keep original in unmasked regions
            scaled_noisy_latent_image = scaled_noisy_latent_image * (1 - binary_mask) \
                                      + scaled_latent_image * binary_mask
            
            print(f"[FlowMatchingMixin] After masking - noisy image range: {scaled_noisy_latent_image.min():.3f} to {scaled_noisy_latent_image.max():.3f}")

        return scaled_noisy_latent_image.to(dtype=orig_dtype), sigmas
