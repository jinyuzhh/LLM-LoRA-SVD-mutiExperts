import enum
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_lora_linear():
    repo_root = Path(__file__).resolve().parent

    peft_pkg = types.ModuleType("peft")
    peft_pkg.__path__ = [str(repo_root / "peft")]
    peft_pkg.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None, is_package=True)
    tuners_pkg = types.ModuleType("peft.tuners")
    tuners_pkg.__path__ = [str(repo_root / "peft" / "tuners")]
    tuners_pkg.__spec__ = importlib.machinery.ModuleSpec("peft.tuners", loader=None, is_package=True)

    class PeftType(str, enum.Enum):
        LORA = "LORA"

    @dataclass
    class PeftConfig:
        base_model_name_or_path: str = None
        peft_type: PeftType = None
        task_type: str = None
        inference_mode: bool = False

    def transpose(weight, fan_in_fan_out):
        return weight.T if fan_in_fan_out else weight

    utils_module = types.ModuleType("peft.utils")
    utils_module.PeftConfig = PeftConfig
    utils_module.PeftType = PeftType
    utils_module.transpose = transpose

    bnb_module = types.ModuleType("bitsandbytes")
    bnb_nn_module = types.ModuleType("bitsandbytes.nn")

    class Linear8bitLt(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias)

    class Linear4bit(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias)

    bnb_nn_module.Linear8bitLt = Linear8bitLt
    bnb_nn_module.Linear4bit = Linear4bit
    bnb_module.nn = bnb_nn_module

    transformers_module = types.ModuleType("transformers")
    transformers_module.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None, is_package=True)
    pytorch_utils_module = types.ModuleType("transformers.pytorch_utils")
    pytorch_utils_module.Conv1D = nn.Linear

    original_find_spec = importlib.util.find_spec

    def find_spec(name, package=None):
        if name == "bitsandbytes":
            return importlib.machinery.ModuleSpec(name, loader=None)
        return original_find_spec(name, package)

    sys.modules["peft"] = peft_pkg
    sys.modules["peft.tuners"] = tuners_pkg
    sys.modules["peft.utils"] = utils_module
    sys.modules["bitsandbytes"] = bnb_module
    sys.modules["bitsandbytes.nn"] = bnb_nn_module
    sys.modules["transformers"] = transformers_module
    sys.modules["transformers.pytorch_utils"] = pytorch_utils_module
    importlib.util.find_spec = find_spec
    try:
        spec = importlib.util.spec_from_file_location(
            "peft.tuners.lora",
            repo_root / "peft" / "tuners" / "lora.py",
        )
        lora_module = importlib.util.module_from_spec(spec)
        sys.modules["peft.tuners.lora"] = lora_module
        spec.loader.exec_module(lora_module)
    finally:
        importlib.util.find_spec = original_find_spec

    return lora_module.Linear


def main():
    Linear = load_lora_linear()
    torch.manual_seed(7)

    batch_size = 2
    seq_len = 3
    in_features = 5
    out_features = 4
    rank = 2

    x = torch.randn(batch_size, seq_len, in_features)

    normal = Linear(
        in_features,
        out_features,
        r=rank,
        lora_alpha=rank,
        lora_nums=1,
        lora_dropout=0.0,
        adaptive=False,
        k=1,
        bias=True,
        adapter_type="lora",
    )
    svd = Linear(
        in_features,
        out_features,
        r=rank,
        lora_alpha=rank,
        lora_nums=1,
        lora_dropout=0.0,
        adaptive=False,
        k=1,
        bias=True,
        adapter_type="svd_lora",
    )

    with torch.no_grad():
        normal.weight.copy_(torch.randn_like(normal.weight))
        normal.bias.copy_(torch.randn_like(normal.bias))
        normal.lora_A0.weight.copy_(torch.randn_like(normal.lora_A0.weight))
        normal.lora_B0.weight.copy_(torch.randn_like(normal.lora_B0.weight))

        svd.weight.copy_(normal.weight)
        svd.bias.copy_(normal.bias)
        svd.lora_A0.weight.copy_(normal.lora_A0.weight)
        svd.lora_B0.weight.copy_(normal.lora_B0.weight)
        svd.lora_svd_e0.fill_(1.0)

    normal.eval()
    svd.eval()

    normal_out = normal(x)
    svd_out = svd(x)

    assert svd_out.shape == (batch_size, seq_len, out_features)
    assert torch.allclose(svd_out, normal_out, atol=1e-6), "svd_e=ones should match normal LoRA"
    assert svd.lora_svd_e0.shape == torch.Size([rank])
    assert "lora_svd_e0" in svd.state_dict()

    with torch.no_grad():
        svd.lora_svd_e0.zero_()

    zero_out = svd(x)
    base_out = F.linear(x, svd.weight, bias=svd.bias)

    assert torch.allclose(zero_out, base_out, atol=1e-6), "svd_e=zeros should zero the adapter update"

    print("SVD-LoRA debug checks passed.")


if __name__ == "__main__":
    main()
