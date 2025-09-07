# pip3 install pyannote.audio torch onnx
# python3 pytorch2onnx.py
# pip freeze > requirements.txt
import torch
from pyannote.audio import Model

# Embedding model ONNX export
print("Converting embedding model to ONNX...")
embedding_model = Model.from_pretrained("./model/pytorch_model.bin").eval()
dummy_input_emb = torch.zeros(2, 1, 160000)

torch.onnx.export(
    embedding_model,
    dummy_input_emb,
    "./model/model.onnx",
    do_constant_folding=True,
    input_names=["input_values"],
    output_names=["embeddings"],
    dynamic_axes={
        "input_values": {0: "batch_size", 1: "num_channels", 2: "num_samples"},
        "embeddings": {0: "batch_size"},
    },
)
print("Embedding model converted successfully!")
