import torch
import torch.nn as nn
from thop import profile
import time

class DownSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 
                              kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(x)
        out = torch.cat([conv_out, pool_out], 1)
        out = self.bn(out)
        return self.act(out)

class ESPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation_rates=[1, 2, 4, 8]):
        super().__init__()
        self.proj_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_channels)
        self.act_proj = nn.PReLU()

        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            self.branches.append(
                nn.Conv2d(out_channels // len(dilation_rates), 
                          out_channels // len(dilation_rates),
                          kernel_size=3, stride=stride, 
                          padding=dilation, dilation=dilation, bias=False)
            )
        self.bn_branches = nn.ModuleList(
            [nn.BatchNorm2d(out_channels // len(dilation_rates)) for _ in dilation_rates]
        )
        self.act_branches = nn.ModuleList([nn.PReLU() for _ in dilation_rates])

        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act_out = nn.PReLU()

    def forward(self, x):
        x = self.proj_1x1(x)
        x = self.bn_proj(x)
        x = self.act_proj(x)

        splits = torch.chunk(x, len(self.branches), dim=1)
        outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(splits[i])
            out = self.bn_branches[i](out)
            out = self.act_branches[i](out)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        out = self.bn_out(out)
        out = self.act_out(out)
        return out

class ESPNet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.level1 = DownSampler(3, 32)
        self.level2_0 = ESPBlock(32, 64, stride=2)
        self.level2_1 = ESPBlock(64, 64)
        self.level3_0 = ESPBlock(64, 128, stride=2)
        self.level3_1 = ESPBlock(128, 128)
        self.level3_2 = ESPBlock(128, 128)

        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.level1(x)
        x = self.level2_0(x)
        x = self.level2_1(x)
        x = self.level3_0(x)
        x = self.level3_1(x)
        x = self.level3_2(x)
        x = self.classifier(x)
        return x

# Instantiate model and input
model = ESPNet(num_classes=27).eval().to('cuda')
input_tensor = torch.randn(1, 3, 720, 1280).to('cuda')

# Parameter and FLOP calculation using thop
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
flops = 2 * macs
print(f"Parameters: {params:,}")
print(f"FLOPs: {flops:,}")

# Inference time measurement
with torch.no_grad():
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    # Timed forward pass
    start = time.time()
    _ = model(input_tensor)
    elapsed = (time.time() - start)  # ms
print(f"Inference time: {elapsed:} ms")
