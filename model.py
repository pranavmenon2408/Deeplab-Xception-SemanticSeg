import torch
import torch.nn as nn
from torch.nn import functional as F
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
    
class Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,dilation=1,exit_flow=False,use_first_relu=True):
        super(Block,self).__init__()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else: self.skip =None
        rep=[]
        self.relu=nn.ReLU(inplace=False)

        rep.append(self.relu)
        rep.append(SeperableConv2D(in_channels,out_channels,3,stride=1,dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeperableConv2D(out_channels,out_channels,3,stride=1,dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeperableConv2D(out_channels,out_channels,3,stride=stride,dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6]=rep[:3]
            rep[:3]=[
                self.relu,
                SeperableConv2D(in_channels,in_channels,3,stride=1,dilation=dilation),
                nn.BatchNorm2d(in_channels)
            ]
        if not use_first_relu: rep=rep[1:]
        self.rep=nn.Sequential(*rep)
    
    def forward(self,x):
        output=self.rep(x)
        if self.skip is not None:
            skip=self.skip(x)
            skip=self.skipbn(skip)
        else: skip=x
        x=output+skip
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
        return x
    
class Decoder(nn.Module):
    def __init__(self,low_level_channels,num_classes):
        super(Decoder,self).__init__()
        self.conv1=nn.Conv2d(low_level_channels,48,1,bias=False)
        self.bn1=nn.BatchNorm2d(48)
        self.relu=nn.ReLU(inplace=False)

        self.last_conv=nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
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
        x=self.last_conv(x)
        return x
    
class DeepLabV3(nn.Module):
    def __init__(self,num_classes=19,output_stride=16):
        super(DeepLabV3,self).__init__()
        self.xception=Xception(output_stride)
        self.aspp=ASPP(2048,output_stride)
        self.decoder=Decoder(128,num_classes)

    def forward(self,x):
        H,W=x.size(2),x.size(3)
        x,low_level_features=self.xception(x)
        x=self.aspp(x)
        x=self.decoder(x,low_level_features)
        x=F.interpolate(x,size=(H,W),mode='bilinear',align_corners=True)
        return x