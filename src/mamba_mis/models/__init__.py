import importlib
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .network import MambaMIS

BASELINES = {
    "unet": ("legacy.unet", "U_Net", {"in_ch": 3, "out_ch": 1}),
    "attunet": ("legacy.AttU_Net", "AttU_Net", {"img_ch": 3, "output_ch": 1}),
    "unetpp": ("cbim.unetpp", "UNetPlusPlus", {"in_ch": 3, "num_classes": 1}),
    "unetv2": ("legacy.unetv2", "UNetV2", {"n_classes": 1, "deep_supervision": False, "pretrained_path": None}),
    "malunet": ("legacy.malunet", "MALUNet", {}),
    "utnetv2": ("cbim.utnetv2", "UTNetV2", {"in_chan": 3, "num_classes": 1}),
    "transfuse": ("transfuse.TransFuse", "TransFuse_S", {"pretrained": False}),
    "sanet": ("sanet.model", "Model", {"args": SimpleNamespace(snapshot=None)}),
    "pranet": ("pranet.PraNet_Res2Net", "PraNet", {}),
    "vmunet": ("legacy.vmunet", "VMUNet", {"load_ckpt_path": None}),
    "vmunetv2": ("legacy.vmunet_v2", "VMUNetV2", {"load_ckpt_path": None, "deep_supervision": False}),
    "hvmunet": ("legacy.H_vmunet", "H_vmunet", {}),
    "ulvmunet": ("legacy.UL_vmunet", "UltraLight_VM_UNet", {}),
}
MODEL_NAMES = tuple(["mamba_mis_s", "mamba_mis_b", "mamba_mis_l", *BASELINES])


class BaselineAdapter(nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model, self.name = model, name

    def forward(self, x):
        result = self.model(x)
        if isinstance(result, (list, tuple)):
            # PraNet: last reverse-attention output; TransFuse: final BiFusion output.
            primary = result[-1] if self.name in {"pranet", "transfuse"} else result[0]
            aux = [y for y in result if y is not primary]
        else:
            primary, aux = result, []
        resize = lambda y: F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False) if y.shape[-2:] != x.shape[-2:] else y
        return {"logits": resize(primary), "aux": [resize(y) for y in aux]}


def build_model(config):
    options = dict(config)
    name = options.pop("name")
    if name.startswith("mamba_mis_"):
        return MambaMIS(variant=name.rsplit('_', 1)[1], **options)
    if name not in BASELINES:
        raise ValueError(f"Unknown model {name}; choose from {MODEL_NAMES}")
    module, symbol, defaults = BASELINES[name]
    kwargs = dict(defaults)
    size = options.pop("image_size", 256)
    # The scan backend of a baseline follows its original Mamba-1 definition.
    options.pop("backend", None)
    options.pop("use_checkpoint", None)
    if name == "transfuse":
        kwargs["image_size"] = size
    kwargs.update(options)
    cls = getattr(importlib.import_module(f".vendor.{module}", __package__), symbol)
    return BaselineAdapter(cls(**kwargs), name)


def logits_of(output):
    return output["logits"] if isinstance(output, dict) else output
