# from mmcv.runner import BaseModule
# from mmseg.models.builder import BACKBONES
# import timm

# @BACKBONES.register_module()
# class TimmBackbone(BaseModule):
#     """timm backbone wrapper for mmseg 0.x (features_only)."""

#     def __init__(self,
#                  model_name,
#                  pretrained=True,
#                  frozen=True,
#                  out_indices=(0, 1, 2, 3),
#                  init_cfg=None,
#                  **kwargs):
#         super().__init__(init_cfg)
#         self.model = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             features_only=True,
#             out_indices=out_indices,
#             **kwargs
#         )
#         if frozen:
#             self.freeze()

#     def freeze(self):
#         """Freeze backbone parameters."""
#         for p in self.model.parameters():
#             p.requires_grad = False

#     def forward(self, x):
#         return self.model(x)


# from mmcv.runner import BaseModule
# from mmseg.models.builder import BACKBONES
# import timm

# @BACKBONES.register_module()
# class TimmBackbone(BaseModule):
#     """timm backbone wrapper for mmseg 0.x (features_only)."""

#     def __init__(self,
#                  model_name,
#                  pretrained=True,
#                  frozen=True,
#                  unfreeze_stage_ids=(),
#                  out_indices=(0, 1, 2, 3),
#                  init_cfg=None,
#                  **kwargs):
#         super().__init__(init_cfg)

#         self.model = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             features_only=True,
#             out_indices=out_indices,
#             **kwargs
#         )

#         if frozen:
#             self.freeze_all()
#             if len(unfreeze_stage_ids) > 0:
#                 self.unfreeze_by_stage_ids(unfreeze_stage_ids)

#     def freeze_all(self):
#         for p in self.model.parameters():
#             p.requires_grad = False

#     def unfreeze_by_stage_ids(self, stage_ids):
#         stage_ids = tuple(stage_ids)
#         patterns = []
#         for i in stage_ids:
#             patterns += [f"stages_{i}.", f"stages.{i}."]

#         matched = 0
#         for name, p in self.model.named_parameters():
#             if any(ptn in name for ptn in patterns):
#                 p.requires_grad = True
#                 matched += 1

#         if matched == 0:
#             sample = [n for n, _ in list(self.model.named_parameters())[:50]]
#             raise AssertionError(
#                 f"Unfreeze matched 0 params for stage_ids={stage_ids}. "
#                 f"First param names: {sample[:10]}"
#             )

#     def forward(self, x):
#         return self.model(x)


from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES
import timm
import torch.nn as nn


@BACKBONES.register_module()
class TimmBackbone(BaseModule):
    """timm backbone wrapper for mmseg 0.x (features_only)."""

    def __init__(self,
                 model_name,
                 pretrained=True,
                 frozen=True,
                 unfreeze_stage_ids=(),
                 freeze_norm=False,                 # <--- NEW
                 norm_types=("layernorm",),         # <--- NEW ("layernorm", "batchnorm", "groupnorm")
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            **kwargs
        )

        self.freeze_norm = freeze_norm
        self.norm_types = tuple(t.lower() for t in norm_types)

        if frozen:
            self.freeze_all()
            if len(unfreeze_stage_ids) > 0:
                self.unfreeze_by_stage_ids(unfreeze_stage_ids)

        # Apply after freezing/unfreezing so norms can remain frozen if desired
        if self.freeze_norm:
            self.freeze_norm_layers()

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_by_stage_ids(self, stage_ids):
        stage_ids = tuple(stage_ids)
        patterns = []
        for i in stage_ids:
            patterns += [f"stages_{i}.", f"stages.{i}."]

        matched = 0
        for name, p in self.model.named_parameters():
            if any(ptn in name for ptn in patterns):
                p.requires_grad = True
                matched += 1

        if matched == 0:
            sample = [n for n, _ in list(self.model.named_parameters())[:50]]
            raise AssertionError(
                f"Unfreeze matched 0 params for stage_ids={stage_ids}. "
                f"First param names: {sample[:10]}"
            )

    def _iter_norm_modules(self):
        """Yield normalization modules according to norm_types."""
        want_ln = "layernorm" in self.norm_types
        want_bn = "batchnorm" in self.norm_types
        want_gn = "groupnorm" in self.norm_types

        for m in self.model.modules():
            if want_ln and isinstance(m, nn.LayerNorm):
                yield m
            if want_bn and isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                yield m
            if want_gn and isinstance(m, nn.GroupNorm):
                yield m

    def freeze_norm_layers(self):
        """
        Freeze norm affine params (LN/GN/BN).
        For BN additionally set eval() to stop running stat updates.
        """
        for m in self._iter_norm_modules():
            # freeze affine params (if present)
            for p in m.parameters(recurse=False):
                p.requires_grad = False

            # BN: prevent running_mean/var updates
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def train(self, mode=True):
        """
        Keep BN in eval mode if freeze_norm is enabled.
        This prevents model.train() from re-enabling BN updates.
        """
        super().train(mode)
        if mode and self.freeze_norm:
            for m in self._iter_norm_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        return self

    def forward(self, x):
        return self.model(x)
