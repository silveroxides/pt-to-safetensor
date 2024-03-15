import torch
from safetensors.torch import load_file, save_file
loaded = torch.load("v2.pt", map_location="cpu")
save_file(loaded, "v2.safetensors", metadata={"format": "pt"})