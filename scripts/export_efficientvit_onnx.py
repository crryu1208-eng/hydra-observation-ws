"""Export ade20k-efficientvit_seg_l2.onnx with semantic_inference's ExportModel (headless).

Weights: https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_ade20k.pt
-> ~/hydra_ws/.models/src/. Run from ~/hydra_ws (paths below are container mounts):

  docker run --rm -e MPLBACKEND=Agg -v $PWD/scripts:/export:ro \
    -v $PWD/src/semantic_inference/exporting:/si_exporting:ro -v $PWD/.models:/models \
    python:3.12-slim bash -c "
      apt-get update -qq && apt-get install -y -qq git >/dev/null
      pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
      pip install -q onnx==1.17.0 onnxsim onnxruntime opencv-python-headless tqdm torchprofile \
        scipy omegaconf torchmetrics==1.4.3 click imageio matplotlib 'numpy<2' einops timm==0.9.16 \
        huggingface_hub ml_dtypes triton git+https://github.com/facebookresearch/segment-anything.git
      pip install -q --no-deps git+https://github.com/mit-han-lab/efficientvit@de7d7733cc0329f391b33f1f459271562ec27bd5
      cd /tmp && python /export/export_efficientvit_onnx.py && chown $(id -u):$(id -g) /models/*.onnx"

Output goes to ~/hydra_ws/.models (set SEMANTIC_INFERENCE_MODEL_DIR=/root/hydra_ws/.models in the
hydra-dev container).
"""

import sys

import efficientvit.seg_model_zoo as zoo
import torch

# current efficientvit renamed create_seg_model(name, dataset, weight_url)
if not hasattr(zoo, "create_seg_model"):
    zoo.create_seg_model = lambda name, dataset, weight_url: zoo.create_efficientvit_seg_model(
        f"efficientvit-seg-{name}-{dataset}", pretrained=True, weight_url=weight_url
    )

sys.path.insert(0, "/si_exporting")
import torch.nn.functional as F  # noqa: E402
from export_efficientvit import ExportModel, _export_legacy  # noqa: E402


class FixedExportModel(ExportModel):
    """Upstream forward resizes back to 512x512 (img is overwritten) and returns int64;
    the C++ Model requires an int32 label tensor and the ROS node does not resize labels."""

    def forward(self, img):
        h, w = img.shape[2], img.shape[3]
        ret = self.model(F.interpolate(img, size=(512, 512), mode="bilinear"))
        ret = F.interpolate(ret, size=(h, w), mode="bilinear")
        return torch.argmax(ret, dim=1).to(torch.int32)


model = FixedExportModel("/models/src/efficientvit_seg_l2_ade20k.pt", name="l2", dataset="ade20k")
model.eval()
img = torch.randn(1, 3, 480, 640, dtype=torch.float32)
with torch.no_grad():
    _export_legacy(model, img, "/models/ade20k-efficientvit_seg_l2.onnx")
    pred = model(img)
print("exported; test output", tuple(pred.shape), pred.dtype, "classes", pred.unique().numel())
