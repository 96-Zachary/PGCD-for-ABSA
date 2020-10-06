#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import netron

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = F.relu(self.block1(x) + identity)
        x = self.output(x)
        return x


d = torch.rand(1, 3, 416, 416)
m = model()
o = m(d)

onnx_path = "onnx_model_name.onnx"
torch.onnx.export(m, d, onnx_path)

netron.start(onnx_path)