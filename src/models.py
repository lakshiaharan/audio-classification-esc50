import torch
import torch.nn as nn
import torchvision.models as tvm


def make_resnet_backbone(pretrained=True, freeze_early=True):
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = tvm.resnet18(weights=weights)

    # spectrograms are 1 channel, resnet expects 3 -> avg the pretrained filters down
    old_conv = net.conv1
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    net.conv1 = new_conv

    feat_dim = net.fc.in_features
    net.fc = nn.Identity()

    if freeze_early:
        for name, param in net.named_parameters():
            if name.startswith(("layer1", "layer2", "conv1", "bn1")):
                param.requires_grad = False

    return net, feat_dim


class SingleResCNN(nn.Module):
    # baseline - single mid-res mel spectrogram, pretrained resnet18
    def __init__(self, n_classes=50, pretrained=True):
        super().__init__()
        self.backbone, feat_dim = make_resnet_backbone(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, x):
        if x.shape[1] == 3:
            # Channel 0 contains the primary static Mel spectrogram
            x = x[:, 0:1, :, :]
        feat = self.backbone(x)
        return self.classifier(feat)


class TimeFreqAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.query = nn.Conv2d(channels, max(channels // reduction, 1), 1)
        self.key = nn.Conv2d(channels, max(channels // reduction, 1), 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, C, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class ResBranch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone, feat_dim = make_resnet_backbone(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.attn = TimeFreqAttention(feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = feat_dim

    def forward(self, x):
        f = self.stem(x)
        f = self.attn(f)
        return self.pool(f).flatten(1)


class MultiResAttentionNet(nn.Module):
    # fine/mid/coarse mel branches, each pretrained resnet18 + attention, fused with
    # a small gating network instead of just concatenating
    def __init__(self, n_classes=50, pretrained=True):
        super().__init__()
        self.branches = nn.ModuleDict({
            name: ResBranch(pretrained=pretrained) for name in ["fine", "mid", "coarse"]
        })
        branch_dim = self.branches["fine"].out_dim
        self.fusion_gate = nn.Sequential(
            nn.Linear(branch_dim * 3, 3),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(branch_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        fine = self.branches["fine"](x[:, 0:1])
        mid = self.branches["mid"](x[:, 1:2])
        coarse = self.branches["coarse"](x[:, 2:3])

        concat = torch.cat([fine, mid, coarse], dim=1)
        weights = torch.softmax(self.fusion_gate(concat), dim=1)

        stacked = torch.stack([fine, mid, coarse], dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        return self.classifier(fused)

    def get_fusion_weights(self, x):
        # lets you check which resolution the model leans on per sample
        with torch.no_grad():
            fine = self.branches["fine"](x[:, 0:1])
            mid = self.branches["mid"](x[:, 1:2])
            coarse = self.branches["coarse"](x[:, 2:3])
            concat = torch.cat([fine, mid, coarse], dim=1)
            weights = torch.softmax(self.fusion_gate(concat), dim=1)
        return weights


class TimeFreqCoordAttention(nn.Module):
    """Decoupled 1D Coordinate Attention across Time and Frequency axes."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h_feat, x_w_feat = torch.split(y, [h, w], dim=2)
        x_w_feat = x_w_feat.permute(0, 1, 3, 2)
        return identity * torch.sigmoid(self.conv_h(x_h_feat)) * torch.sigmoid(self.conv_w(x_w_feat))


class MultiFeatureCoordNet(nn.Module):
    """
    SOTA Multi-Feature Differential Attention Network:
    Takes 3-channel input: [Static Mel (64 mels), 1st Delta (Velocity), 2nd Delta (Acceleration)].
    Uses full 3-channel pretrained ResNet18 + Time-Frequency Coordinate Attention.
    """
    def __init__(self, n_classes=50, pretrained=True, freeze_early=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)
        self.conv1 = resnet.conv1 # Native 3-channel RGB ImageNet filters
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.attn3 = TimeFreqCoordAttention(256)
        self.attn4 = TimeFreqCoordAttention(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
        if freeze_early:
            for m in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attn3(self.layer3(x))
        x = self.attn4(self.layer4(x))
        feat = self.pool(x).flatten(1)
        return self.classifier(feat)

