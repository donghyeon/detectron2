import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import CNNBlockBase

from .resnet import (
    BasicBlock,
    BottleneckBlock,
    DeformBottleneckBlock,
    BasicStem,
    ResNet,
    make_stage
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY


class AttentionBlock(CNNBlockBase):
    """Multi-headed attention layer."""

    def __init__(self, in_channels, out_channels, *,
                 bottleneck_channels=None, num_heads=1, attention_dropout=0.9,
                 stride=1, norm="LN"):
        super().__init__(in_channels, out_channels, stride)

        self.in_channels = in_channels
        if bottleneck_channels is None:
            bottleneck_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = nn.Linear(in_channels, bottleneck_channels, bias=False)
        self.k_dense_layer = nn.Linear(in_channels, bottleneck_channels, bias=False)
        self.v_dense_layer = nn.Linear(in_channels, bottleneck_channels, bias=False)
        self.output_dense_layer = nn.Linear(bottleneck_channels, out_channels, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = None

        self.layer_norm = nn.LayerNorm(in_channels)

        # Randomly initializing parameters
        for layer in [self.q_dense_layer, self.k_dense_layer, self.v_dense_layer,
                      self.output_dense_layer, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)
                # nn.init.normal_(layer, std=0.001)

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        Args:
            x: A tensor with shape [batch_size, length, bottleneck_channels]
        Returns:
            A tensor with shape [batch_size, num_heads, length, bottleneck_channels/num_heads]
        """
        batch_size, length, _ = x.size()

        # Calculate depth of last dimension after it has been split.
        depth = (self.bottleneck_channels // self.num_heads)

        # Split the last dimension
        x = torch.reshape(x, [batch_size, length, self.num_heads, depth])

        # Transpose the result
        return torch.transpose(x, 1, 2)

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
            x: A tensor [batch_size, num_heads, length, bottleneck_channels/num_heads]
        Returns:
            A tensor with shape [batch_size, length, bottleneck_channels]
        """
        batch_size, _, length, _ = x.size()
        x = torch.transpose(x, 1, 2)  # --> [batch, length, num_heads, depth]
        return torch.reshape(x, [batch_size, length, self.bottleneck_channels])

    def forward(self, x, y):
        """Apply attention mechanism to x and y.
        Args:
            x: a tensor with shape [batch_size, length_x, bottleneck_channels]
            y: a tensor with shape [batch_size, length_y, bottleneck_channels]
            bias: attention bias that will be added to the result of the dot product.
            cache: (Used during prediction) dictionary with tensors containing results
                of previous attentions. The dictionary must have the items:
                    {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.
        Returns:
            Attention layer output with shape [batch_size, length_x, bottleneck_channels]
        """
        # Preprocessing
        x = self.layer_norm(x)
        y = self.layer_norm(y)

        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.bottleneck_channels // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = torch.matmul(q, torch.transpose(k, 2, 3))
        weights = F.softmax(logits)
        # TODO: Check why dropout hurts the evaluation. For now, dropout is disabled.
        # weights = self.dropout_layer(weights)
        out = torch.matmul(weights, v)

        # Recombine heads --> [batch_size, length, bottleneck_channels]
        out = self.combine_heads(out)

        # Run the combined outputs through another linear projection layer.
        out = self.output_dense_layer(out)

        # Postprocessing: apply dropout and residual connection
        # TODO: Check why dropout hurts the evaluation. For now, dropout is disabled.
        # out = self.dropout_layer(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = self.layer_norm(out)
        return out


class SelfAttentionBlock(AttentionBlock):
    def forward(self, x):
        return super().forward(x, x)


class SelfAttention2DBlock(SelfAttentionBlock):
    def forward(self, x):
        batch, depth, height, width = x.size() # NCHW
        x = x.permute(0, 2, 3, 1) # NHWC
        x = torch.reshape(x, [batch, height * width, depth]) # NLC
        x = super().forward(x)
        x = torch.reshape(x, [batch, height, width, depth]) # NHWC
        x = x.permute(0, 3, 1, 2) # NCHW
        return x


class ResNetStages(object):
    def __init__(self, stages):
        """A helper class to append custom blocks on ResNet stages

        Args:
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
        """
        self.stages = stages
        for i, stage in enumerate(stages):
            # Starting from res2
            setattr(self, "res{}".format(i + 2), stage)

    def append_blocks(self, block_class, num_blocks, *, stage="res4", **kwargs):
        """Append custom blocks on the target ResNet stage
        Args:
            block_class (type): a subclass of CNNBlockBase
            num_blocks (int):
            stage (str): the target ResNet stage to append
            kwargs (dict): A dict of arguments for block_class
        """
        target_stage = getattr(self, stage)
        for i in range(num_blocks):
            target_stage.append(block_class(**kwargs))


class AttResNetStages(ResNetStages):
    def append_self_attention_blocks(self, stage="res4", num_heads=4):
        in_channels = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}[stage]

        kwargs = {}
        kwargs["in_channels"] = in_channels
        kwargs["out_channels"] = in_channels
        kwargs["bottleneck_channels"] = int(in_channels / 4)
        kwargs["num_heads"] = num_heads
        self.append_blocks(SelfAttention2DBlock, num_blocks=1, stage=stage, **kwargs)


@BACKBONE_REGISTRY.register()
def build_attresnet_backbone(cfg, input_shape):
    """
        Create a ResNet instance from config.

        Returns:
            ResNet: a :class:`ResNet` instance.
        """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS

    attention_stages = cfg.MODEL.RESNETS.ATTENTION.STAGES
    attention_heads = cfg.MODEL.RESNETS.ATTENTION.HEADS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)

    # Append attention block
    attresnet_stages = AttResNetStages(stages)
    for attention_stage, attention_head in zip(attention_stages, attention_heads):
        attresnet_stages.append_self_attention_blocks(stage=attention_stage,
                                                      num_heads=attention_head)
    stages = attresnet_stages.stages

    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)
