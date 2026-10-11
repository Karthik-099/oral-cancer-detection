"""
Export OralNet-B4 for in-browser inference (ONNX Runtime Web).

    python -m inference.export_onnx --out ../models/oralnet-b4.onnx

The exported graph returns both the logits and a Grad-CAM map, so the browser
needs no autograd. That works because the classifier head is a small MLP on
globally pooled features: the gradient of the (suspicious - normal) margin
with respect to the pooled features can be written out in closed form, and
Grad-CAM's channel weights are exactly that gradient divided by H*W (a
positive constant that normalisation removes).
"""
import argparse
import hashlib
import logging

import numpy as np
import torch
import torch.nn as nn

from .engine import OralNetB4
from .registry import find_weights

log = logging.getLogger("oralaid.export")
INPUT_SIZE = 380


def _fold_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    """Fold eval-mode BatchNorm into the preceding Linear layer."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = linear.weight * scale[:, None]
    b = (linear.bias - bn.running_mean) * scale + bn.bias
    return w.detach(), b.detach()


class ExplainableOralNet(nn.Module):
    def __init__(self, model: OralNetB4):
        super().__init__()
        self.features = model.backbone.features
        head = model.backbone.classifier
        w1, b1 = _fold_bn(head[1], head[2])
        w2, b2 = _fold_bn(head[5], head[6])
        self.register_buffer("w1", w1)
        self.register_buffer("b1", b1)
        self.register_buffer("w2", w2)
        self.register_buffer("b2", b2)
        self.register_buffer("w3", head[9].weight.detach())
        self.register_buffer("b3", head[9].bias.detach())

    def forward(self, x):
        a = self.features(x)  # (B, 1792, h, w)
        pooled = a.mean(dim=(2, 3))
        z1 = pooled @ self.w1.T + self.b1
        h1 = torch.relu(z1)
        z2 = h1 @ self.w2.T + self.b2
        h2 = torch.relu(z2)
        logits = h2 @ self.w3.T + self.b3

        # d(logit_suspicious - logit_normal) / d(pooled), by hand.
        margin = (self.w3[1] - self.w3[0]).unsqueeze(0)  # (1, 256)
        g2 = margin * (z2 > 0).float()
        g1 = (g2 @ self.w2) * (z1 > 0).float()
        g0 = g1 @ self.w1  # (B, 1792)
        cam = torch.relu((g0[:, :, None, None] * a).sum(dim=1))  # (B, h, w)
        return logits, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../models/oralnet-b4.onnx")
    args = ap.parse_args()

    path = find_weights("oralnet-b4")
    if not path:
        raise SystemExit("OralNet-B4 weights not found")
    model = OralNetB4()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    wrapped = ExplainableOralNet(model).eval()

    dummy = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        wrapped, dummy, args.out, opset_version=17,
        input_names=["image"], output_names=["logits", "cam"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}},
        do_constant_folding=True,
    )

    # Parity check against the PyTorch model on a real-looking input.
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    logits, cam = sess.run(None, {"image": x.numpy()})
    err = float(np.abs(ref - logits).max())
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"exported {args.out}  max|logit diff|={err:.2e}  cam={cam.shape}  sha256={digest}")
    if err > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch")


if __name__ == "__main__":
    main()
