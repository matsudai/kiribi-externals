# pip3 install pyannote.audio torch onnx
# python3 pytorch2onnx.py
# pip freeze > requirements.txt
import torch
from pyannote.audio import Model

# Segmentation model ONNX export
print("Converting segmentation model to ONNX...")
segmentation_model = Model.from_pretrained("./model/pytorch_model.bin").eval()
dummy_input_seg = torch.zeros(2, 1, 160000)

torch.onnx.export(
    segmentation_model,
    dummy_input_seg,
    "./model/model.onnx",
    do_constant_folding=True,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
        "logits": {0: "batch_size", 1: "num_frames"},
    },
)
print("Segmentation model converted successfully!")
