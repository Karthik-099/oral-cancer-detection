"""
U-Net Segmentation Model
Lesion boundary segmentation for oral cancer images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.decoders.append(ConvBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final(x))


@lru_cache(maxsize=1)
def load_segmenter(model_path, device):
    model = UNet()
    if model_path and __import__('os').path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"✅ Segmenter loaded from {model_path}")
    else:
        print("⚠️  Segmenter using random weights (not trained)")
    return model.to(device).eval()


def get_segmentation_mask(model, image_tensor, device, threshold=0.5):
    with torch.no_grad():
        inp = F.interpolate(image_tensor, size=(256, 256))
        mask = model(inp.to(device))
        mask = (mask > threshold).float()
    return mask.squeeze().cpu().numpy()
