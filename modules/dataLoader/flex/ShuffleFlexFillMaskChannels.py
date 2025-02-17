from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class ShuffleFlexFillMaskChannels(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(self, in_name: str, out_name: str):
        super().__init__()
        self.in_name = in_name
        self.out_name = out_name

    def length(self) -> int:
        return self._get_previous_length(self.in_name)

    def get_inputs(self) -> list[str]:
        return [self.in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        mask = self._get_previous_item(variation, self.in_name, index)

        height, width = mask.shape[1], mask.shape[2]
        vae_scale_factor = 8

        # Reshape to match Flex model's expected dimensions [16, 56, 56]
        # First reshape to [height, 8, width, 8]
        mask = mask.view(
            height // vae_scale_factor,
            vae_scale_factor,
            width // vae_scale_factor,
            vae_scale_factor,
        )
        # Then permute to [8, 8, height, width]
        mask = mask.permute(1, 3, 0, 2)
        # Finally reshape to [16, height, width] to match Flex's axes_dims_rope[0]
        mask = mask.reshape(
            16,  # Fixed to match Flex model's expected dimension
            height // vae_scale_factor,
            width // vae_scale_factor,
        )

        return {
            self.out_name: mask
        } 