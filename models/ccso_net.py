from tracemalloc import start
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from thop import profile
import time


class LSCFEM(nn.Module):
    """Long-Short Configurable Context Feature Enhancement Module"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True)
        )
        
        # Short-range context branch
        self.short_ctx = nn.Sequential(
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3, 
                     padding=1, groups=in_channels//reduction, bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True)
        )
        
        # Long-range context branch
        self.long_ctx = nn.Sequential(
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3,
                     padding=2, dilation=2, groups=in_channels//reduction, bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(2*(in_channels//reduction), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        reduced = self.channel_reduction(x)
        short = self.short_ctx(reduced)
        long = self.long_ctx(reduced)
        combined = torch.cat([short, long], dim=1)
        attention = self.fusion(combined)
        return x * attention + x

class SOADM(nn.Module):
    """Small Object Attention Decoding Module"""
    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, 1, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adjust attention layer to expect 2 * high_channels
        self.attention = nn.Sequential(
            nn.Conv2d(2 * high_channels, high_channels, 1, bias=False),
            nn.BatchNorm2d(high_channels),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        low_feat = F.interpolate(low_feat, size=high_feat.shape[2:], mode='bilinear', align_corners=True)
        low_proj = self.low_proj(low_feat)
        combined = torch.cat([low_proj, high_feat], dim=1)
        attn_map = self.attention(combined)
        return high_feat * attn_map + low_proj


class CCSONet(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()
        # Backbone initialization (ResNet18)
        backbone = resnet18(pretrained=pretrained)
        self.initial = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels

        # Context modules
        self.lscfem3 = LSCFEM(256)  # Processes layer3 (256 channels)
        self.lscfem4 = LSCFEM(512)  # Processes layer4 (512 channels)

        # Decoding modules (FIXED HERE)
        self.soadm43 = SOADM(256, 512)  # Processes layer3 (256) and layer4 (512)
        self.soadm32 = SOADM(128, 512)  # Processes layer2 (128) and upsampled d3 (512)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x, target_size=(720, 1280)):
        # Encoder
        x0 = self.initial(x)    # 1/4 resolution
        x1 = self.layer1(x0)    # 1/4 (64 channels)
        x2 = self.layer2(x1)    # 1/8 (128 channels)
        x3 = self.lscfem3(self.layer3(x2))  # 1/16 (256 channels)
        x4 = self.lscfem4(self.layer4(x3))  # 1/32 (512 channels)

        # Decoder
        d4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)  # 1/16
        d3 = self.soadm43(x3, d4)  

        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 1/8
        d2 = self.soadm32(x2, d3)  

        # Final upsampling and prediction
        out = self.head(d2)  # Reduce channels BEFORE upsampling
        if target_size is None:
            out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        return out


# Test the implementation
if __name__ == "__main__":
    model = CCSONet(num_classes=27).to('cpu')
    x = torch.randn(1, 3, 720, 1280).to("cpu")  # Batch of 2, 512x1024 images
    start_time = time.time()
    with torch.no_grad():
        output = model(x)
    original_time = time.time() - start_time
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be (2, 19, 512, 1024)
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameters: {params / 1e6:.2f} M")
    print(f"Original inference time: {original_time:.4f} seconds")
