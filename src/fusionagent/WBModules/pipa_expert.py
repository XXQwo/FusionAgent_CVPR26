import torch
from torch import nn
from torchvision.models import resnet50


class PIPAResNetExpert(nn.Module):
    """ResNet-50 cue expert returning a 2048-D embedding."""

    output_dim = 2048

    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)
        self.features = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


def load_pipa_expert(checkpoint_path):
    model = PIPAResNetExpert()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("backbone_state_dict", checkpoint.get("model_state_dict", checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "PIPA expert checkpoint mismatch. missing={} unexpected={}".format(
                missing, unexpected
            )
        )
    model.eval()
    return model
