from turtle import back
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from torchvision.models import mobilenet_v3_large


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

        self.medium_ctx = nn.Sequential(
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3,
                        padding=2, dilation=2, groups=in_channels//reduction, bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True)
        )
        
        # Long-range context branch
        self.long_ctx = nn.Sequential(
            nn.Conv2d(in_channels//reduction, in_channels//reduction, 3,
                     padding=4, dilation=4, groups=in_channels//reduction, bias=False),
            nn.BatchNorm2d(in_channels//reduction),
            nn.ReLU(inplace=True)
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3*(in_channels//reduction), (in_channels//reduction)//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((in_channels//reduction)//4, 3*(in_channels//reduction), 1, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(3*(in_channels//reduction), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        reduced = self.channel_reduction(x)
        short = self.short_ctx(reduced)
        medium = self.medium_ctx(reduced)
        long = self.long_ctx(reduced)

        combined = torch.cat([short, medium, long], dim=1)
        
        ca_weights = self.channel_attention(combined)
        combined = combined * ca_weights
        attention = self.fusion(combined)
        return x * attention + x

class MultiLevelFeatureFusion(nn.Module):
    """Enhanced feature fusion with multiple low-level feature extraction points"""
    def __init__(self, backbone_channels=[96, 192, 384], aspp_channels=64):
        super().__init__()

        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Multiple low-level feature processors
        self.low_level_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ) for channels in backbone_channels
        ])

        self.boundary_attention = nn.Sequential(
           nn.Conv2d(32 * len(backbone_channels) + 1, 32 * len(backbone_channels), 3, padding=1, bias=False),
           nn.BatchNorm2d(32 * len(backbone_channels)),    
           nn.Sigmoid()
        )
        
        # Channel reduction after attention
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(32 * len(backbone_channels), 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Feature pyramid fusion with LSCFEM
        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(48 + aspp_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            LSCFEM(128),  # Use existing LSCFEM for context enhancement
            nn.Dropout(0.1)
        )
        
    def forward(self, high_features, low_level_features_list, input_image=None):
        # Process multiple low-level features
        processed_low = []
        target_size = low_level_features_list[0].shape[2:]
        #print([features.shape for features in low_level_features_list])
        
        for i, (processor, features) in enumerate(zip(self.low_level_processors, low_level_features_list)):
            processed = processor(features)
            if processed.shape[2:] != target_size:
                processed = F.interpolate(processed, size=target_size, mode='bilinear', align_corners=True)
            processed_low.append(processed)
        
        # Cross-scale attention - FIXED
        combined_low = torch.cat(processed_low, dim=1)  # Shape: [B, 96, H, W]
                # Generate boundary information if input image is provided
        if input_image is not None:
            edge_map = self.edge_detector(input_image)
            edge_map = F.interpolate(edge_map, size=target_size, mode='bilinear', align_corners=True)
           
            # Combine with edge information for boundary-aware attention
            combined_with_edge = torch.cat([combined_low, edge_map], dim=1)
            attention_weights = self.boundary_attention(combined_with_edge)
            attended_low = combined_low * attention_weights
        else:
            attended_low = combined_low
       
        # Reduce channels
        attended_low = self.channel_reduction(attended_low)
       
        # Upsample high-level features
        high_upsampled = F.interpolate(high_features, size=target_size, mode='bilinear', align_corners=True)
       
        # Final fusion
        fused = torch.cat([attended_low, high_upsampled], dim=1)
        output = self.pyramid_fusion(fused)





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
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, groups=4, bias=False)
        
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
    Lighter Xception backbone with multiple feature extraction points
    """
    def __init__(self, output_stride=16, in_channels=3):
        super(LiteXception, self).__init__()
        if output_stride == 16: 
            b3_s, mf_d, ef_d = 2, 1, (1, 2)
        if output_stride == 8: 
            b3_s, mf_d, ef_d = 1, 2, (2, 4)

        # Initial convolutions
        self.conv1 = nn.Conv2d(3, 24, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(24, 48, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        # Entry flow blocks with reduced channels
        self.block1 = Block(48, 96, stride=2, dilation=1, use_first_relu=False)
        self.ccar1 = CCAR(96)
        self.block2 = Block(96, 192, stride=2, dilation=1)
        self.ccar2 = CCAR(192)
        self.block3 = Block(192, 384, stride=b3_s, dilation=1)
        self.ccar3 = CCAR(384)

        # Middle flow
        self.midflow = nn.Sequential(
            *[Block(384, 384, stride=1, dilation=mf_d) for _ in range(4)]
        )

        #self.ccar_midflow = CCAR(384)
        
        # Exit flow
        self.block4 = Block(384, 512, stride=1, dilation=ef_d[0], exit_flow=True)
        # Final separable convolutions
        self.conv3 = AdaptiveContextDSConv(512, 768, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]])  
        self.bn3 = nn.BatchNorm2d(768)
        self.conv4 = AdaptiveContextDSConv(768, 768, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]])  
        self.bn4 = nn.BatchNorm2d(768)
        self.conv5 = AdaptiveContextDSConv(768, 1024, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]]) 
        self.bn5 = nn.BatchNorm2d(1024)
        #self.ccar_final = CCAR(1024)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Extract multiple low-level features
        x = self.block1(x)
        x = self.ccar1(x)
        low_level_1 = x  # 96 channels
        
        x = F.relu(x)
        x = self.block2(x)
        x = self.ccar2(x)
        low_level_2 = x  # 192 channels
        
        x = self.block3(x)
        x = self.ccar3(x)
        low_level_3 = x  # 384 channels

        # Middle flow
        x = self.midflow(x)
        #x = self.ccar_midflow(x)

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
        #x = self.ccar_final(x)

        return x, [low_level_1, low_level_2, low_level_3]
    
class MobileNetV3_Enhanced(nn.Module):
    """
    Enhanced MobileNetV3 backbone with multiple feature extraction points
    (Direct replacement for LiteXception)
    """
    def __init__(self, output_stride=16, in_channels=3):
        super(MobileNetV3_Enhanced, self).__init__()
        
        # Load pretrained MobileNetV3
        backbone = mobilenet_v3_large(pretrained=True)
        self.features = backbone.features
        
        # Define feature extraction points to match LiteXception's pattern
        # LiteXception extracts at: 96, 192, 384 channels
        # MobileNetV3 equivalent points: after layers 3, 6, 12
        self.extraction_points = [3, 6, 12]
        self.feature_channels = [24, 40, 112]  # Corresponding channel counts
        
        # Add CCAR modules at feature extraction points
        self.ccar1 = CCAR(24)   # After layer 3
        self.ccar2 = CCAR(40)   # After layer 6  
        self.ccar3 = CCAR(112)  # After layer 12
        
        # Replace final layers with AdaptiveContextDSConv (matching LiteXception pattern)
        if output_stride == 16: 
            ef_d = (1, 2)
        if output_stride == 8: 
            ef_d = (2, 4)
            
        #Final separable convolutions (replacing original final layers)
        self.conv3 = AdaptiveContextDSConv(960, 768, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]])  
        self.bn3 = nn.BatchNorm2d(768)
        self.conv4 = AdaptiveContextDSConv(768, 768, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]])  
        self.bn4 = nn.BatchNorm2d(768)
        self.conv5 = AdaptiveContextDSConv(768, 1024, kernel_size=3, stride=1, dilation_rates=[1, ef_d[1]]) 
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Extract multiple low-level features (exactly like LiteXception)
        low_level_features = []
        
        # Process through MobileNetV3 layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Extract features at specific points with CCAR enhancement
            if i == 3:  # First extraction point
                x = self.ccar1(x)
                low_level_1 = x  # 40 channels (equivalent to LiteXception's 96)
                low_level_features.append(low_level_1)
            elif i == 6:  # Second extraction point
                x = self.ccar2(x)
                low_level_2 = x  # 80 channels (equivalent to LiteXception's 192)
                low_level_features.append(low_level_2)
            elif i == 12:  # Third extraction point
                x = self.ccar3(x)
                low_level_3 = x  # 160 channels (equivalent to LiteXception's 384)
                low_level_features.append(low_level_3)
        
        # Apply final AdaptiveContextDSConv layers (like LiteXception's exit flow)
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

        # Return in exact same format as LiteXception: (final_features, [low_level_1, low_level_2, low_level_3])
        return x, low_level_features



class HierarchicalMSASPP(nn.Module):
    """Enhanced Multi-Scale ASPP with better context modeling"""
    def __init__(self, in_channels, output_stride=16):
        super().__init__()
        
        if output_stride == 16:
            dilations = [1, 6, 12, 18]  # Standard dilations for better coverage
        else:
            dilations = [1, 12, 24, 36]
        
        reduced_channels = 64
        
        # Multi-scale branches with depthwise separable convolutions
        self.ms_branches = nn.ModuleList()
        
        # 1x1 conv branch
        self.ms_branches.append(nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(reduced_channels, 8), num_channels=reduced_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated branches with depthwise separable convolutions
        for dil in dilations[1:]:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=dil, 
                         dilation=dil, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
                nn.GroupNorm(num_groups=min(reduced_channels, 8), num_channels=reduced_channels),
                nn.ReLU(inplace=True)
            )
            self.ms_branches.append(branch)
        
        # Pyramid pooling with proper channel calculation
        self.pyramid_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(4)
        ])
        
        # Make sure pyramid channels add up to exactly reduced_channels
        pyramid_channels = [16, 24, 24]  # 22 + 21 + 21 = 64
        self.pyramid_convs = nn.ModuleList()
        for channels in pyramid_channels:
            self.pyramid_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, bias=False),
                nn.GroupNorm(num_groups=min(channels, 8), num_channels=channels),  # Use GroupNorm
                nn.ReLU(inplace=True)
            ))
        
        # Feature fusion with attention
        total_channels = reduced_channels * 5  # 4 ASPP + 1 global = 320 channels
        self.feature_attention = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 4, total_channels, 1),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, reduced_channels, 1, bias=False),
            nn.GroupNorm(num_groups=min(reduced_channels, 8), num_channels=reduced_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        ms_features = [branch(x) for branch in self.ms_branches]
        
        # Global pyramid pooling
        global_features = []
        for pool, conv in zip(self.pyramid_pooling, self.pyramid_convs):
            pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = F.interpolate(conv_out, size=x.shape[2:], mode='bilinear', align_corners=True)
            global_features.append(upsampled)
        
        global_combined = torch.cat(global_features, dim=1)  # Will be exactly 64 channels
        
        # Combine all features
        all_features = ms_features + [global_combined]
        combined = torch.cat(all_features, dim=1)
        
        # Apply attention
        attention = self.feature_attention(combined)
        attended = combined * attention
        
        # Final processing
        output = self.final_conv(attended)
        
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
    
class EnhancedDecoder(nn.Module):
    def __init__(self, backbone_channels=[96, 192, 384], num_classes=19, aspp_channels=64):
        super().__init__()
        
        # Multi-level feature fusion
        self.feature_fusion = MultiLevelFeatureFusion(backbone_channels, aspp_channels)
        
        # Keep existing CCAR for enhanced attention
        #self.ccar_decoder = CCAR(128)
        
        # Final classification layers
        self.last_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, 1, stride=1),
        )

    def forward(self, high_features, low_level_features_list):
        # Multi-level feature fusion
        fused = self.feature_fusion(high_features, low_level_features_list)
        
        # Apply CCAR attention
        #fused = self.ccar_decoder(fused)
        
        # Final classification
        output = self.last_conv(fused)
        
        return output
    
    
class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder with progressive upsampling"""
    def __init__(self, backbone_channels=[24, 40, 112], num_classes=19, aspp_channels=64):
        super().__init__()
        
        # Progressive upsampling decoders
        self.decoder_4x = nn.Sequential(
            nn.Conv2d(aspp_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.decoder_2x = nn.Sequential(
            nn.Conv2d(128 + backbone_channels[2], 64, 3, padding=1, bias=False),  # 112 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.decoder_1x = nn.Sequential(
            nn.Conv2d(64 + backbone_channels[1], 32, 3, padding=1, bias=False),  # 40 channels
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Deep supervision outputs for training
        self.aux_classifier_4x = nn.Conv2d(128, num_classes, 1)
        self.aux_classifier_2x = nn.Conv2d(64, num_classes, 1)

    def adaptive_concat(self, feat1, feat2, dim=1):
        """Concatenate features with automatic size matching"""
        if feat1.shape[2:] != feat2.shape[2:]:
            # Resize feat2 to match feat1's spatial dimensions
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], 
                                mode='bilinear', align_corners=True)
        return torch.cat([feat1, feat2], dim=dim)

    def forward(self, aspp_features, low_level_features_list, return_aux=False):
        # Debug prints (remove after fixing)
        
        # Progressive upsampling with skip connections
        # 4x upsampling
        x = self.decoder_4x(aspp_features)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        aux_4x = self.aux_classifier_4x(x) if return_aux else None
        
        # 2x upsampling with skip connection
        x = self.adaptive_concat(x, low_level_features_list[2])
        
        x = self.decoder_2x(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        aux_2x = self.aux_classifier_2x(x) if return_aux else None
        
        # 1x with skip connection
        x = self.adaptive_concat(x, low_level_features_list[1])
        
        x = self.decoder_1x(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        
        # Final classification
        x = self.classifier(x)
        
        
        if return_aux:
            return x, aux_4x, aux_2x
        return x
    
class LiteDeepLabV3(nn.Module):
    """Improved LiteDeepLabV3 with better accuracy while maintaining efficiency"""
    def __init__(self, num_classes=19, output_stride=16, use_mobilenet_v3=True):
        super().__init__()
        
        if use_mobilenet_v3:
            self.backbone = MobileNetV3_Enhanced(output_stride)
        else:
            self.backbone = LiteXception(output_stride)
        
        # Enhanced ASPP
        self.aspp = HierarchicalMSASPP(1024, output_stride)
        
        # Multi-scale decoder
        self.decoder = MultiScaleDecoder([24, 40, 112], num_classes, aspp_channels=64)
        
        # Boundary-aware feature fusion
        self.boundary_fusion = MultiLevelFeatureFusion([24, 40, 112], aspp_channels=64)

    def forward(self, x, return_aux=False):
        H, W = x.size(2), x.size(3)
        
        # Extract features
        backbone_features, low_level_features_list = self.backbone(x)
        
        # Enhanced ASPP
        aspp_features = self.aspp(backbone_features)
        
        # Multi-scale decoding
        if return_aux and self.training:
            output, aux_4x, aux_2x = self.decoder(aspp_features, low_level_features_list, return_aux=True)
            
            # Upsample auxiliary outputs
            aux_4x = F.interpolate(aux_4x, size=(H, W), mode='bilinear', align_corners=True)
            aux_2x = F.interpolate(aux_2x, size=(H, W), mode='bilinear', align_corners=True)
            
            # Final upsampling
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
            
            return output, aux_4x, aux_2x
        else:
            output = self.decoder(aspp_features, low_level_features_list, return_aux=False)
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
            return output 

    
from thop import profile

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Initialize models with both ASPP variants
    #model_original_aspp = LiteDeepLabV3(num_classes=num_classes, output_stride=16, use_hierarchical_aspp=False, use_mobilenet_v3=False).to(device)
    model_hierarchical_aspp = LiteDeepLabV3(num_classes=num_classes, output_stride=16, use_mobilenet_v3=True).to(device)
    
    # Set models to evaluation mode
    #model_original_aspp.eval()
    model_hierarchical_aspp.eval()
    
    # Print model summaries
    #original_params = sum(p.numel() for p in model_original_aspp.parameters())
    hierarchical_params = sum(p.numel() for p in model_hierarchical_aspp.parameters())
    
    #print(f"LiteDeepLabV3 with Original ASPP parameters: {original_params:,}")
    print(f"LiteDeepLabV3 with HierarchicalMSASPP parameters: {hierarchical_params:,}")
    #print(f"Parameter reduction with HierarchicalMSASPP: {(1 - hierarchical_params/original_params)*100:.2f}%")
    
    # Forward pass with timing for both models
    # start_time = time.time()
    # with torch.no_grad():
    #     output_original = model_original_aspp(x)
    # original_time = time.time() - start_time
    
    start_time = time.time()
    with torch.no_grad():
        output_hierarchical = model_hierarchical_aspp(x)
    hierarchical_time = time.time() - start_time
    
    # Print output information
    #print(f"Original ASPP output shape: {output_original.shape}")
    print(f"HierarchicalMSASPP output shape: {output_hierarchical.shape}")
    #print(f"Original ASPP forward pass time: {original_time:.4f} seconds")
    print(f"HierarchicalMSASPP forward pass time: {hierarchical_time:.4f} seconds")
    #print(f"Speed improvement: {(original_time/hierarchical_time - 1)*100:.2f}%")

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         model_hierarchical_aspp(x)
    # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # # Apply dynamic quantization to conv layers
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model_hierarchical_aspp,
    #     {nn.Conv2d},
    #     dtype=torch.qint8
    # )

    # # Must run calibration
    # with torch.no_grad():
    #     for _ in range(10):
    #         quantized_model(torch.randn(1,3,720,1280))

    # start_time = time.time()
    # with torch.no_grad():
    #     output_hierarchical = quantized_model(x)
    # hierarchical_time = time.time() - start_time
    # print(f"Quantized HierarchicalMSASPP forward pass time: {hierarchical_time:.4f} seconds")

    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU],
    #     record_shapes=True
    # ) as prof:
    #     quantized_model(x)
    # print(prof.key_averages().table(sort_by="cpu_time_total"))




    
    # Optional: Calculate and print memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_hierarchical_aspp(x)
        hierarchical_mem = torch.cuda.max_memory_allocated()/1024**2
        
        torch.cuda.reset_peak_memory_stats()
        # with torch.no_grad():
        #     _ = model_original_aspp(x)
        # original_mem = torch.cuda.max_memory_allocated()/1024**2
        
        # print(f"Original ASPP GPU Memory: {original_mem:.2f} MB")
        # print(f"HierarchicalMSASPP GPU Memory: {hierarchical_mem:.2f} MB")
        # print(f"Memory reduction: {(1 - hierarchical_mem/original_mem)*100:.2f}%")

        # flops_original, params_original = profile(model_original_aspp, inputs=(x,))
        flops_hierarchical, params_hierarchical = profile(model_hierarchical_aspp, inputs=(x,))
        #print(f"Original ASPP FLOPs: {flops_original/1e9:.2f} GFLOPs, Params: {params_original/1e6:.2f} M")
        print(f"HierarchicalMSASPP FLOPs: {flops_hierarchical/1e9:.2f} GFLOPs, Params: {params_hierarchical/1e6:.2f} M")
        #print(f"FLOPs reduction with HierarchicalMSASPP: {(1 - flops_hierarchical/flops_original)*100:.2f}%")

if __name__ == "__main__":
    main()