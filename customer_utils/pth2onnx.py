import torch
import torch.nn as nn
from model.deeplab import DeepLab

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = DeepLab(backbone='resnet')
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("../ckp/model59_0.42959582026409104.pth", map_location=device))
model.eval()
tensor_test = torch.randn((1, 3, 244, 244), device=device)
print(tensor_test.shape)
torch.onnx._export(model, tensor_test, "model99.onnx", verbose=True, opset_version=11)