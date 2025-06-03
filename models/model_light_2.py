import torch
import torch.nn as nn
from torch.nn import functional as F
import time

class CCAR(nn.Module):
    """
    Optimized Conditional Channel-wise Attention Routing (CCAR) module
    """
    def __init__(self, channels, reduction=16, spatial_reduction=8):
        super(CCAR, self).__init__()
        
        # Parameter reduction by increasing reduction ratio
        reduced_channels = max(8, channels // reduction)
        
        # Global context branch - shared MLP for parameter efficiency
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        
        # Local context branch - spatial attention (simplified)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, groups=1, bias=False),
            nn.Sigmoid()
        )
        
        # Conditional routing branch (simplified)
        self.routing_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Use group convolution for fusion to reduce parameters
        self.fusion_conv = nn.Conv2d(channels * 2, channels, kernel_size=1, groups=2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global context path - channel attention
        global_context = self.global_pool(x)
        global_weight = self.shared_mlp(global_context)
        
        # Local context path - spatial attention (simplified)
        local_context = self.local_conv(x)
        
        # Conditional routing weights (simplified)
        routing_weights = self.routing_fc(x)
        
        # Apply routing weights - create two attention paths
        path1 = global_weight * routing_weights[:, 0:1, :, :]
        path2 = global_weight * routing_weights[:, 1:2, :, :]
        
        # Combine with spatial attention
        path1 = path1 * local_context
        path2 = path2 * (1 - local_context)
        
        # Feature refinement
        refined1 = x * path1
        refined2 = x * path2
        
        # Feature fusion
        combined = torch.cat([refined1, refined2], dim=1)
        output_weight = self.fusion_conv(combined)
        output_weight = self.sigmoid(output_weight)
        
        return x * output_weight

class SeperableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeperableConv2D, self).__init__()
        if dilation > kernel_size//2: 
            padding = dilation
        else: 
            padding = kernel_size//2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AdaptiveContextDSConv(nn.Module):
    """
    Optimized Adaptive Context-Aware Depthwise Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation_rates=[1, 2]):
        super(AdaptiveContextDSConv, self).__init__()
        
        # Reduce parameter count by using fewer dilation rates
        self.dilation_rates = dilation_rates
        
        # Create depthwise convolutions with different dilation rates
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
        
        # Simplified context encoder with fewer parameters
        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(dilation_rates), kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Pointwise convolution with group convolution for parameter efficiency
        groups = 1 if in_channels < 32 or out_channels < 32 else min(4, min(in_channels, out_channels) // 16)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        
        # Normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
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
        
        # Apply batch norm and activation
        output = self.bn(output)
        output = self.relu(output)
        
        return output

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_first_relu=True, use_adaptive_conv=True):
        super(Block, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else: 
            self.skip = None
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
        if not use_first_relu: 
            rep = rep[1:]
        self.rep = nn.Sequential(*rep)
    
    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else: 
            skip = x
        x = output + skip
        return x

class LiteXception(nn.Module):
    """
    Lighter Xception backbone with reduced channels
    """
    def __init__(self, output_stride=16, in_channels=3):
        super(LiteXception, self).__init__()
        if output_stride == 16: 
            b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8: 
            b3_s, mf_d, ef_d = 1, 2, (2, 4)

        # Initial convolutions
        self.conv1 = nn.Conv2d(3, 24, 3, 2, padding=1, bias=False)  # Reduced from 32
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(24, 48, 3, 1, padding=1, bias=False)  # Reduced from 64
        self.bn2 = nn.BatchNorm2d(48)

        # Entry flow blocks with reduced channels
        self.block1 = Block(48, 96, stride=2, dilation=1, use_first_relu=False)  # Reduced from 128
        self.block2 = Block(96, 192, stride=2, dilation=1)  # Reduced from 256
        self.block3 = Block(192, 384, stride=b3_s, dilation=1)  # Reduced from 728

        # Middle flow - reduced number of blocks and channels
        self.midflow = nn.Sequential(
            *[Block(384, 384, stride=1, dilation=mf_d) for _ in range(8)]  # Reduced from 16 blocks and 728 channels
        )

        # Exit flow
        self.block4 = Block(384, 512, stride=1, dilation=ef_d[0], exit_flow=True)  # Reduced from 1024

        # Final separable convolutions
        self.conv3 = AdaptiveContextDSConv(512, 768, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]])  
        self.bn3 = nn.BatchNorm2d(768)
        self.conv4 = AdaptiveContextDSConv(768, 768, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]])  
        self.bn4 = nn.BatchNorm2d(768)
        self.conv5 = AdaptiveContextDSConv(768, 1024, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]]) 
        self.bn5 = nn.BatchNorm2d(1024)
    
    def forward(self, x):
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
        self.ccar = CCAR(reduced_channels)
    
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
        
        # Apply CCAR attention
        output = self.ccar(output)
        
        return output

# Keep the old ASPP for backward compatibility
class Asppbranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Asppbranch, self).__init__()
        padding = 0 if kernel_size == 1 else dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)
    
class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride=16):
        super(ASPP, self).__init__()
        if output_stride == 16: 
            dilations = [1, 6, 12, 18]
        if output_stride == 8: 
            dilations = [1, 12, 24, 36]

        reduced_channels = 128  # Reduced from 256

        self.branch1 = Asppbranch(in_channels, reduced_channels, 1, dilations[0])
        self.branch2 = Asppbranch(in_channels, reduced_channels, 3, dilations[1])
        self.branch3 = Asppbranch(in_channels, reduced_channels, 3, dilations[2])
        self.branch4 = Asppbranch(in_channels, reduced_channels, 3, dilations[3])

        self.ccar = CCAR(reduced_channels)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=False)
        )

        # Use group convolution in the final 1x1 to reduce parameters
        self.conv1 = nn.Conv2d(reduced_channels * 5, reduced_channels, 1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.avgpool(x)
        b5 = F.interpolate(b5, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        x = torch.cat((b1, b2, b3, b4, b5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ccar(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, aspp_channels=64):
        super(Decoder, self).__init__()
        # Updated for HierarchicalMSASPP output channels
        #aspp_channels = 64  # Updated to match HierarchicalMSASPP output
        
        self.conv1 = nn.Conv2d(low_level_channels, 32, 1, bias=False)  # Reduced from 48
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        self.ccar_decoder = CCAR(32 + aspp_channels)  # Updated for reduced channels

        # More efficient decoder
        self.last_conv = nn.Sequential(
            nn.Conv2d(32 + aspp_channels, 128, 3, stride=1, padding=1, groups=1, bias=False),  # Reduced from 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, 1, stride=1),
        )

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat((low_level_features, x), dim=1)
        x = self.ccar_decoder(x)
        x = self.last_conv(x)
        return x
    
class LiteDeepLabV3(nn.Module):
    def __init__(self, num_classes=19, output_stride=16, use_hierarchical_aspp=True):
        super(LiteDeepLabV3, self).__init__()
        self.xception = LiteXception(output_stride)
        
        # Choose between HierarchicalMSASPP and original ASPP
        if use_hierarchical_aspp:
            self.aspp = HierarchicalMSASPP(1024, output_stride)
            self.decoder = Decoder(96, num_classes, aspp_channels=64)  # Reduced from 128
        else:
            self.aspp = ASPP(1024, output_stride)
            self.decoder = Decoder(96, num_classes, aspp_channels=128)  # Original ASPP uses 128 channels
            
          # Reduced from 128

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.xception(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
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
    input_height = 256
    input_width = 256
    num_classes = 26
    
    # Create random input tensor
    x = torch.randn(batch_size, input_channels, input_height, input_width).to(device)
    print(f"Input tensor shape: {x.shape}")
    
    # Initialize models with both ASPP variants
    model_original_aspp = LiteDeepLabV3(num_classes=num_classes, output_stride=16, use_hierarchical_aspp=False).to(device)
    model_hierarchical_aspp = LiteDeepLabV3(num_classes=num_classes, output_stride=16, use_hierarchical_aspp=True).to(device)
    
    # Set models to evaluation mode
    model_original_aspp.eval()
    model_hierarchical_aspp.eval()
    
    # Print model summaries
    original_params = sum(p.numel() for p in model_original_aspp.parameters())
    hierarchical_params = sum(p.numel() for p in model_hierarchical_aspp.parameters())
    
    print(f"LiteDeepLabV3 with Original ASPP parameters: {original_params:,}")
    print(f"LiteDeepLabV3 with HierarchicalMSASPP parameters: {hierarchical_params:,}")
    print(f"Parameter reduction with HierarchicalMSASPP: {(1 - hierarchical_params/original_params)*100:.2f}%")
    
    # Forward pass with timing for both models
    start_time = time.time()
    with torch.no_grad():
        output_original = model_original_aspp(x)
    original_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        output_hierarchical = model_hierarchical_aspp(x)
    hierarchical_time = time.time() - start_time
    
    # Print output information
    print(f"Original ASPP output shape: {output_original.shape}")
    print(f"HierarchicalMSASPP output shape: {output_hierarchical.shape}")
    print(f"Original ASPP forward pass time: {original_time:.4f} seconds")
    print(f"HierarchicalMSASPP forward pass time: {hierarchical_time:.4f} seconds")
    print(f"Speed improvement: {(original_time/hierarchical_time - 1)*100:.2f}%")

    
    # Optional: Calculate and print memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_hierarchical_aspp(x)
        hierarchical_mem = torch.cuda.max_memory_allocated()/1024**2
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_original_aspp(x)
        original_mem = torch.cuda.max_memory_allocated()/1024**2
        
        print(f"Original ASPP GPU Memory: {original_mem:.2f} MB")
        print(f"HierarchicalMSASPP GPU Memory: {hierarchical_mem:.2f} MB")
        print(f"Memory reduction: {(1 - hierarchical_mem/original_mem)*100:.2f}%")

        flops_original, params_original = profile(model_original_aspp, inputs=(x,))
        flops_hierarchical, params_hierarchical = profile(model_hierarchical_aspp, inputs=(x,))
        print(f"Original ASPP FLOPs: {flops_original/1e9:.2f} GFLOPs, Params: {params_original/1e6:.2f} M")
        print(f"HierarchicalMSASPP FLOPs: {flops_hierarchical/1e9:.2f} GFLOPs, Params: {params_hierarchical/1e6:.2f} M")
        print(f"FLOPs reduction with HierarchicalMSASPP: {(1 - flops_hierarchical/flops_original)*100:.2f}%")

if __name__ == "__main__":
    main()