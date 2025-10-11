import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
from torchvision.models import mobilenet_v3_large

class MobileNetV3Backbone(nn.Module):
    def __init__(self, output_stride=16, pretrained=True):
        super().__init__()
        # Load pretrained MobileNetV3 Large
        backbone = mobilenet_v3_large(pretrained=pretrained)
        self.features = backbone.features
        
        # Remove unused final layers
        self.features = nn.Sequential(*list(self.features.children())[:-1])
        
        # Feature extraction points
        self.low_level_features_idx = 3  # After 4th layer (stride 4)
        self.high_level_features_idx = -1  # Last feature map
        
        # Output stride adaptation
        if output_stride == 16:
            self._set_dilation(2)
    
    def _set_dilation(self, dilation_rate):
        """Adjust stride to dilation for output stride 16"""
        for i in range(16, len(self.features)):
            module = self.features[i]
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                    module.dilation = (dilation_rate, dilation_rate)
                    padding = dilation_rate
                    module.padding = (padding, padding)
    
    def forward(self, x):
        low_level_feat = None
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == self.low_level_features_idx:
                low_level_feat = x  # Shape: [N, 40, H/4, W/4]
        
        high_level_feat = x  # Shape: [N, 960, H/16, W/16]
        return high_level_feat, low_level_feat
    

class CCAR(nn.Module):
    """
    Conditional Channel-wise Attention Routing (CCAR) module
    
    This module implements a more sophisticated attention mechanism that:
    1. Captures both spatial and channel dependencies
    2. Uses multiple attention paths for conditional routing
    3. Combines global and local context information
    """
    def __init__(self, channels, reduction=16, spatial_reduction=8):
        super(CCAR, self).__init__()
        
        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        
        # Local context branch (spatial attention)
        self.local_conv = nn.Conv2d(channels, channels // spatial_reduction, kernel_size=3, padding=1, bias=False)
        self.local_bn = nn.BatchNorm2d(channels // spatial_reduction)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_conv2 = nn.Conv2d(channels // spatial_reduction, 1, kernel_size=1, bias=False)
        
        # Conditional routing branch
        self.routing_pool = nn.AdaptiveAvgPool2d(1)
        self.routing_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 2, kernel_size=1, bias=False)  # 2 routing paths
        )
        
        # Final fusion layer
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Global context path - channel attention
        global_context = self.global_pool(x)
        global_weight = self.global_fc1(global_context)
        global_weight = self.global_relu(global_weight)
        global_weight = self.global_fc2(global_weight)
        
        # Local context path - spatial attention
        local_context = self.local_conv(x)
        local_context = self.local_bn(local_context)
        local_context = self.local_relu(local_context)
        local_context = self.local_conv2(local_context)
        
        # Conditional routing weights
        routing_weights = self.routing_pool(x)
        routing_weights = self.routing_fc(routing_weights)
        routing_weights = F.softmax(routing_weights, dim=1)
        
        # Apply routing weights - create two attention paths
        path1 = global_weight * routing_weights[:, 0:1, :, :]
        path2 = global_weight * routing_weights[:, 1:2, :, :]
        
        # Combine with spatial attention
        path1 = path1 * torch.sigmoid(local_context)
        path2 = path2 * (1 - torch.sigmoid(local_context))
        
        # Feature refinement
        refined1 = x * path1
        refined2 = x * path2
        
        # Feature fusion
        combined = torch.cat([refined1, refined2], dim=1)
        output_weight = self.fusion_conv(combined)
        output_weight = self.sigmoid(output_weight)
        
        # Apply final attention weights
        output = x * output_weight
        
        return output

# # [Original code implementation from the paste.txt file]
# class CCARGate(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(CCARGate, self).__init__()
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         context = self.global_pool(x)
#         weight = self.fc1(context)
#         weight = self.relu(weight)
#         weight = self.fc2(weight)
#         weight = self.sigmoid(weight)
#         return x * weight


class SeperableConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,dilation=1,bias=False):
        super(SeperableConv2D,self).__init__()
        if dilation > kernel_size//2: padding = dilation
        else: padding = kernel_size//2
        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,bias=bias)

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AdaptiveContextDSConv(nn.Module):
    """
    Adaptive Context-Aware Depthwise Separable Convolution
    
    This module dynamically adjusts convolution parameters based on input features:
    1. Adapts dilation rates based on feature context
    2. Uses position-sensitive depth multipliers
    3. Integrates local pattern recognition with global context
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation_rates=[1, 2, 3]):
        super(AdaptiveContextDSConv, self).__init__()
        
        self.dilation_rates = dilation_rates
        
        # Create multiple depthwise convolutions with different dilation rates
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=dilation * (kernel_size//2), 
                dilation=dilation, 
                groups=in_channels, 
                bias=False
            ) for dilation in dilation_rates
        ])
        
        # Context encoder to predict dilation weights
        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, len(dilation_rates), kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Pointwise convolution to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Channel-wise scaling factors
        self.channel_scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Get context-based weights for different dilation rates
        weights = self.context_encoder(x)
        
        # Apply each dilated depthwise convolution
        dilated_outputs = []
        for i, conv in enumerate(self.depthwise_convs):
            dilated_output = conv(x)
            # Extract the weight for this dilation rate
            weight = weights[:, i:i+1, :, :].expand_as(dilated_output)
            dilated_outputs.append(dilated_output * weight)
        
        # Sum the weighted dilated outputs
        depthwise_output = sum(dilated_outputs)
        
        # Apply pointwise convolution
        output = self.pointwise(depthwise_output)
        
        # Apply channel-wise scaling
        output = output * self.channel_scale
        
        # Apply batch norm and activation
        output = self.bn(output)
        output = self.relu(output)
        
        return output
    
class HierarchicalMSASPP(nn.Module):
    """
    Multi-Scale ASPP with hierarchical feature aggregation
    """
    def __init__(self, in_channels, output_stride=16):
        super(HierarchicalMSASPP, self).__init__()
        
        if output_stride == 16:
            dilations = [1, 3, 6, 9]  # Smaller, more fine-grained dilations
        else:
            dilations = [1, 6, 12, 18]
        
        reduced_channels = 64
        
        # Multi-scale branches with different kernel sizes
        self.ms_branches = nn.ModuleList()
        kernel_sizes = [1, 3, 5, 7]
        
        for i, (dil, ks) in enumerate(zip(dilations, kernel_sizes)):
            if ks == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                padding = (ks // 2) * dil
                branch = nn.Sequential(
                    # Depthwise separable with multi-scale
                    nn.Conv2d(in_channels, in_channels, ks, padding=padding, 
                             dilation=dil, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
                    nn.BatchNorm2d(reduced_channels),
                    nn.ReLU(inplace=True)
                )
            self.ms_branches.append(branch)
        
        # Hierarchical aggregation - combine features at different levels
        self.level1_fusion = nn.Conv2d(reduced_channels * 2, reduced_channels, 1, bias=False)
        self.level2_fusion = nn.Conv2d(reduced_channels * 2, reduced_channels, 1, bias=False)
        
        # Global context with squeeze and excitation
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels // 4, reduced_channels, 1),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d(reduced_channels * 3, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        
        # Add CCAR for enhanced attention
        #self.ccar = CCAR(reduced_channels)
    
    def forward(self, x):
        # Extract multi-scale features
        ms_features = [branch(x) for branch in self.ms_branches]
        
        # Hierarchical aggregation
        # Level 1: Combine fine-scale features (1x1 and 3x3)
        level1 = self.level1_fusion(torch.cat([ms_features[0], ms_features[1]], dim=1))
        
        # Level 2: Combine medium-scale features (5x5 and 7x7)  
        level2 = self.level2_fusion(torch.cat([ms_features[2], ms_features[3]], dim=1))
        
        # Global context
        global_ctx = self.global_context(x)
        global_features = level1 * global_ctx + level2 * (1 - global_ctx)
        
        # Interpolate global features to match spatial dimensions
        global_features = F.interpolate(global_features, size=x.shape[2:], 
                                      mode='bilinear', align_corners=True)
        
        # Final fusion
        final_features = torch.cat([level1, level2, global_features], dim=1)
        output = self.final_conv(final_features)
        print(f"Output shape after final conv: {output.shape}")
        
        # Apply CCAR attention
        #output = self.ccar(output)
        
        return output
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_first_relu=True, use_adaptive_conv=False):
        super(Block, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else: self.skip = None
        rep = []
        self.relu = nn.ReLU(inplace=False)

        # First block
        rep.append(self.relu)
        if use_adaptive_conv:
            rep.append(AdaptiveContextDSConv(in_channels, out_channels, kernel_size=3, stride=1, dilation_rates=[1, dilation]))
        else:
            rep.append(SeperableConv2D(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        # Second block
        rep.append(self.relu)
        if use_adaptive_conv:
            rep.append(AdaptiveContextDSConv(out_channels, out_channels, kernel_size=3, stride=1, dilation_rates=[1, dilation]))
        else:
            rep.append(SeperableConv2D(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        # Third block
        rep.append(self.relu)
        if use_adaptive_conv:
            rep.append(AdaptiveContextDSConv(out_channels, out_channels, kernel_size=3, stride=stride, dilation_rates=[1, dilation]))
        else:
            rep.append(SeperableConv2D(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                AdaptiveContextDSConv(in_channels, in_channels, kernel_size=3, stride=1, dilation_rates=[1, dilation]) if use_adaptive_conv 
                else SeperableConv2D(in_channels, in_channels, 3, stride=1, dilation=dilation),
                nn.BatchNorm2d(in_channels)
            ]
        if not use_first_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)
    
    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else: skip = x
        x = output + skip
        return x
    
class Xception(nn.Module):
    def __init__(self,output_stride=16,in_channels=3):
        super(Xception,self).__init__()
        if output_stride == 16: b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8: b3_s, mf_d, ef_d = 1, 2, (2, 4)

        self.conv1=nn.Conv2d(3,32,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.relu=nn.ReLU(inplace=False)
        self.conv2=nn.Conv2d(32,64,3,1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(64)

        self.block1=Block(64,128,stride=2,dilation=1,use_first_relu=False)
        self.block2=Block(128,256,stride=2,dilation=1)
        self.block3=Block(256,728,stride=b3_s,dilation=1)

        self.midflow=nn.Sequential(
            *[Block(728,728,stride=1,dilation=mf_d) for _ in range(16)]
        )

        self.block4=Block(728,1024,stride=1,dilation=ef_d[0],exit_flow=True)

        self.conv3=SeperableConv2D(1024,1536,3,stride=1,dilation=ef_d[1])
        self.bn3=nn.BatchNorm2d(1536)
        self.conv4=SeperableConv2D(1536,1536,3,stride=1,dilation=ef_d[1])
        self.bn4=nn.BatchNorm2d(1536)
        self.conv5=SeperableConv2D(1536,2048,3,stride=1,dilation=ef_d[1])
        self.bn5=nn.BatchNorm2d(2048)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        low_level_features = x
        x = F.relu(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)

        # Exit flow
        x = self.block4(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_features

class Asppbranch(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,dilation):
        super(Asppbranch,self).__init__()
        padding = 0 if kernel_size == 1 else dilation
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,dilation=dilation,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self,x):
        return self.conv(x)
    
class ASPP(nn.Module):
    def __init__(self,in_channels,output_stride=16):
        super(ASPP,self).__init__()
        if output_stride == 16: dilations = [1, 6, 12, 18]
        if output_stride == 8: dilations = [1, 12, 24, 36]

        self.branch1=Asppbranch(in_channels,256,1,dilations[0])
        self.branch2=Asppbranch(in_channels,256,3,dilations[1])
        self.branch3=Asppbranch(in_channels,256,3,dilations[2])
        self.branch4=Asppbranch(in_channels,256,3,dilations[3])

        self.ccar = CCAR(256)

        self.avgpool=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_channels,256,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )

        self.conv1=nn.Conv2d(256*5,256,1,bias=False)
        self.bn1=nn.BatchNorm2d(256)
        self.relu=nn.ReLU(inplace=False)
        self.dropout=nn.Dropout(0.5)

    def forward(self,x):
        b1=self.branch1(x)
        b2=self.branch2(x)
        b3=self.branch3(x)
        b4=self.branch4(x)
        b5=self.avgpool(x)
        b5=F.interpolate(b5,size=(x.shape[2],x.shape[3]),mode='bilinear',align_corners=True)

        x=torch.cat((b1,b2,b3,b4,b5),dim=1)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.dropout(x)
        #x = self.ccar(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,low_level_channels,num_classes):
        super(Decoder,self).__init__()
        self.conv1=nn.Conv2d(low_level_channels,48,1,bias=False)
        self.bn1=nn.BatchNorm2d(48)
        self.relu=nn.ReLU(inplace=False)

        #self.ccar_decoder = CCAR(48+256)

        self.last_conv=nn.Sequential(
            nn.Conv2d(48+64, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )

    def forward(self,x,low_level_features):
        low_level_features=self.conv1(low_level_features)
        low_level_features=self.bn1(low_level_features)
        low_level_features=self.relu(low_level_features)

        x=F.interpolate(x,size=(low_level_features.size(2),low_level_features.size(3)),mode='bilinear',align_corners=True)
        x=torch.cat((low_level_features,x),dim=1)
        #x = self.ccar_decoder(x)
        x=self.last_conv(x)
        return x
    
class DeepLabV3(nn.Module):
    def __init__(self,num_classes=19,output_stride=16):
        super(DeepLabV3,self).__init__()
        #self.xception=Xception(output_stride)
        self.xception=MobileNetV3Backbone(output_stride=output_stride)
        #self.aspp=ASPP(160,output_stride)
        self.aspp=HierarchicalMSASPP(160,output_stride=output_stride)
        self.decoder=Decoder(24,num_classes)

    def forward(self,x):
        H,W=x.size(2),x.size(3)
        x,low_level_features=self.xception(x)
        x=self.aspp(x)
        x=self.decoder(x,low_level_features)
        x=F.interpolate(x,size=(H,W),mode='bilinear',align_corners=True)
        return x
    
from thop import profile

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a random input tensor (batch_size, channels, height, width)
    batch_size = 1
    input_channels = 3
    input_height = 720
    input_width = 1280
    num_classes = 27
    
    # Create random input tensor
    x = torch.randn(batch_size, input_channels, input_height, input_width).to(device)
    print(f"Input tensor shape: {x.shape}")
    
    # Initialize the model with output_stride=16 (default)
    model = DeepLabV3(num_classes=num_classes, output_stride=16).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Print model summary (number of parameters)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass with timing
    start_time = time.time()
    with torch.no_grad():
        output = model(x)
    end_time = time.time()
    
    # Print output information
    print(f"Output tensor shape: {output.shape}")
    print(f"Forward pass time: {(end_time - start_time):.4f} seconds")
    
    # Verify expected output shape
    expected_shape = (batch_size, num_classes, input_height, input_width)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
    print("Output shape verification: Passed")
    
    # Optional: Calculate and print memory usage
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

    flops_original, params_original = profile(model, inputs=(x,))
        
    print(f"Original ASPP FLOPs: {flops_original/1e9:.2f} GFLOPs, Params: {params_original/1e6:.2f} M")
def test_ccar():
    # Create a test tensor
    x = torch.randn(2, 256, 64, 64)
    
    # Test the original CCARGate
    #ccar_gate = CCARGate(256)
    #out_gate = ccar_gate(x)
    
    # Test the improved CCAR
    ccar = CCAR(256)
    out_ccar = ccar(x)
    
    print(f"Input shape: {x.shape}")
    #print(f"CCARGate output shape: {out_gate.shape}")
    #print(f"CCAR output shape: {out_ccar.shape}")
    
    return out_ccar

if __name__ == "__main__":
    main()

