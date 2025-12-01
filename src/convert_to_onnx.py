# src/convert_to_onnx.py
import torch
from model import MultiTaskCNN
import os

ONNX_OUTPUT = "models/multitask_cnn.onnx"
PT_MODEL = "models/multitask_cnn.pth"

print("Loading PyTorch model...")
model = MultiTaskCNN(num_emotions=7)
model.load_state_dict(torch.load(PT_MODEL, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 128, 128)  # same size as your CNN

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT,
    input_names=["input"],
    output_names=["emotion", "age", "gender"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}}
)

print("DONE! Saved:", ONNX_OUTPUT)
