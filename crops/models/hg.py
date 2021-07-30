import torch
import torch.nn as nn
import torch.nn.functional as F

## Hourglass implementation - work in progress


class ResBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(feat_in)
        self.conv1 = nn.Conv2d(feat_in, feat_out // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feat_out // 2)
        self.conv2 = nn.Conv2d(feat_out // 2, feat_out // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(feat_out // 2)
        self.conv3 = nn.Conv2d(feat_out // 2, feat_out, kernel_size=1, bias=False)
        
        self.downsample = None
        if feat_in != feat_out:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(feat_in),
                nn.ReLU(True),
                nn.Conv2d(feat_in, feat_out, kernel_size=1, stride=1, bias=False),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out, True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, True)
        out = self.conv2(out)

        out = self.bn3(out)
        out = F.relu(out, True)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        return out

class Hourglass(nn.Module):
    def __init__(self, num_blocks, num_feats, depth):
        super(Hourglass, self).__init__()
        self.num_blocks = num_blocks
        self.num_feats = num_feats
        self.depth = depth

        for d in range(depth):
            self.add_module("upper_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
            self.add_module("lower_in_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))

            # Additional blocks at bottom of hourglass
            if d == 0:
                self.add_module("lower_plus_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
            
            self.add_module("lower_out_branch_" + str(d), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
        
    def _forward(self, depth, inp):
         # Upper branch
        up1 = inp
        up1 = self._modules['upper_branch_' + str(depth)](up1)

        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['lower_in_branch_' + str(depth)](low1)

        if depth > 0:
            low2 = self._forward(depth - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['lower_plus_branch_' + str(depth)](low2)

        low3 = low2
        low3 = self._modules['lower_out_branch_' + str(depth)](low3)

        up2 = F.upsample(low3, scale_factor=2, mode='nearest')

        return up1 + up2
    
    def forward(self, x):
        return self._forward(self.depth - 1, x)


class StackedHourglass(nn.Module):
    def __init__(self, num_stack, num_blocks, num_feats, input_channels, output_channels):
        super(StackedHourglass, self).__init__()
        
        self.num_stack = num_stack
        self.num_blocks = num_blocks
        self.num_feats = num_feats
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Initial filtering
        self.init_conv1 =  nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.init_bn1 = nn.BatchNorm2d(64)
        self.init_res1 = ResBlock(64, 128)
        self.init_res2 = ResBlock(128,128)
        self.init_res3 = ResBlock(128,num_feats)

        for s in range(num_stack):
            # Hourglass
            self.add_module("hg_" + str(s), Hourglass(num_blocks, num_feats, 4))
            # ResBlocks on end at output resolution
            self.add_module("hg_res_" + str(s), nn.Sequential(*[ResBlock(num_feats, num_feats) for i in range(num_blocks)]))
            
            # Final convolutions and BN
            self.add_module("hg_conv1_" + str(s), nn.Conv2d(num_feats, num_feats, kernel_size=1, stride=1, padding=0))
            self.add_module("hg_bn_" + str(s), nn.BatchNorm2d(num_feats))
            self.add_module("hg_l_" + str(s), nn.Conv2d(num_feats, output_channels, kernel_size=1, stride=1, padding=0))
            
            if s < self.num_stack - 1:
                self.add_module('hg_ll_pred_' + str(s), nn.Conv2d(num_feats, num_feats, kernel_size=1, stride=1, padding=0))
                self.add_module('hg_out_pred_' + str(s), nn.Conv2d(output_channels, num_feats, kernel_size=1, stride=1, padding=0))
                
    
    def forward(self, x):
        # Initial Convolutions
        x = self.init_conv1(x)
        x = self.init_bn1(x)
        x = F.relu(x, True)
        x = self.init_res1(x)
        x = F.max_pool2d(x, 2)
        x = self.init_res2(x)
        x = self.init_res3(x)

        previous = x
        output = []

        for i in range(self.num_stack):
            hg = self._modules['hg_' + str(i)](previous)

            ll = hg
            ll = self._modules['hg_res_' + str(i)](ll)

            ll = F.relu(self._modules['hg_bn_' + str(i)](self._modules['hg_conv1_' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['hg_l_' + str(i)](ll)
            output.append(tmp_out)

            if i < self.num_stack - 1:
                ll = self._modules['hg_ll_pred_' + str(i)](ll)
                tmp_out_ = self._modules['hg_out_pred_' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return output