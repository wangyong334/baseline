"""Two-device FP32 U-Net partition; original checkpoint keys are unchanged."""
import torch
import spconv.pytorch as spconv

from model.evspsegnet import evspsegnet


def transfer_sparse(x, device):
    """Move features with autograd, but deliberately do NOT copy indice caches.

    Every downsample/inverse pair stays on one device. At the return boundary
    UR_block_forward uses the original local lateral tensor and its local cache.
    Rebuilding a tensor here must never detach its features.
    """
    device = torch.device(device)
    if x.features.device == device:
        return x
    with torch.cuda.device(device):
        return spconv.SparseConvTensor(
            x.features.to(device), x.indices.to(device),
            x.spatial_shape, x.batch_size,
        )


class evspsegnet_mp(evspsegnet):
    def __init__(self, cfg, split=2):
        if split not in (1, 2, 3):
            raise ValueError("split must be 1, 2 or 3")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Expose TWO GPUs with CUDA_VISIBLE_DEVICES=0,1")
        super().__init__(cfg)
        self.split = split
        self.devices = (torch.device("cuda:0"), torch.device("cuda:1"))
        self.conv_input.to(self.devices[0])
        self.semantic_linear.to(self.devices[0])
        for stage in range(1, 5):
            device = self.stage_device(stage)
            names = ["conv%d" % stage, "conv_up_t%d" % stage,
                     "conv_up_m%d" % stage]
            if stage > 1:
                names += ["pa%d" % stage, "inv_conv%d" % stage]
            else:
                names += ["conv5"]
            for name in names:
                getattr(self, name).to(device)

    def stage_device(self, stage):
        return self.devices[int(stage > self.split)]

    def forward(self, input):
        if input.features.dtype != torch.float32 or torch.is_autocast_enabled():
            raise RuntimeError("This experimental entry requires FP32, without AMP")
        if input.features.device != self.devices[0]:
            raise RuntimeError("Voxelization/input must be on logical cuda:0")
        skips = {}
        with torch.cuda.device(self.devices[0]):
            x = self.conv_input(input)
        for stage in range(1, 5):
            device = self.stage_device(stage)
            with torch.cuda.device(device):
                x = transfer_sparse(x, device)
                x = getattr(self, "conv%d" % stage)(x)
                if stage > 1:
                    x = getattr(self, "pa%d" % stage)(x)
                skips[stage] = x
        for stage in range(4, 0, -1):
            device = self.stage_device(stage)
            with torch.cuda.device(device):
                crossing = x.features.device != device
                x = transfer_sparse(x, device)
                lateral = skips.pop(stage)
                if crossing:
                    # Concatenation assumes exactly the same coordinate order.
                    if (list(x.spatial_shape) != list(lateral.spatial_shape)
                            or x.batch_size != lateral.batch_size
                            or not torch.equal(x.indices, lateral.indices)):
                        raise RuntimeError("Cross-device skip coordinate order mismatch")
                inverse = (getattr(self, "inv_conv%d" % stage)
                           if stage > 1 else self.conv5)
                x = self.UR_block_forward(
                    lateral, x, getattr(self, "conv_up_t%d" % stage),
                    getattr(self, "conv_up_m%d" % stage), inverse,
                )
        with torch.cuda.device(self.devices[0]):
            output = self.semantic_linear(x.features)
            return output, x.replace_feature(output)
