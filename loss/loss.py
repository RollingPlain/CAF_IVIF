import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GradientLoss(nn.Module):
    # 梯度loss
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.L1loss = nn.L1Loss()

    def forward(self, x, s1, s2):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        grad_f = torch.abs(sobelx) + torch.abs(sobely)

        sobelx_s1 = F.conv2d(s1, self.weightx, padding=1)
        sobely_s1 = F.conv2d(s1, self.weighty, padding=1)
        grad_s1 = torch.abs(sobelx_s1) + torch.abs(sobely_s1)

        sobelx_s2 = F.conv2d(s2, self.weightx, padding=1)
        sobely_s2 = F.conv2d(s2, self.weighty, padding=1)
        grad_s2 = torch.abs(sobelx_s2) + torch.abs(sobely_s2)

        grad_max = torch.max(grad_s1, grad_s2)

        loss = self.L1loss(grad_f, grad_max)
        return loss

class GradientLoss2(nn.Module):
    def __init__(self):
        super(GradientLoss2, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.L1loss = nn.L1Loss()

    def forward(self, fus, oth):
        sobelx = F.conv2d(fus, self.weightx, padding=1)
        sobely = F.conv2d(fus, self.weighty, padding=1)
        grad_f = torch.abs(sobelx) + torch.abs(sobely)

        sobelx_s1 = F.conv2d(oth, self.weightx, padding=1)
        sobely_s1 = F.conv2d(oth, self.weighty, padding=1)
        grad_s1 = torch.abs(sobelx_s1) + torch.abs(sobely_s1)

        loss = self.L1loss(grad_f, grad_s1)
        return loss

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(weights=True).features
        self.layer_names = ['0', '3', '8', '15', '22']
        self.layers = nn.ModuleList([nn.Sequential(getattr(self.vgg, l)) for l in self.layer_names])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):

        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        x = self.normalize(x)
        y = self.normalize(y)
        x_feats = self.get_features(x)
        y_feats = self.get_features(y)
        loss = 0
        for xf, yf in zip(x_feats, y_feats):
            loss += torch.mean(torch.abs(xf.detach() - yf.detach()))
        return loss

    def get_features(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

    def normalize(self, x):
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std

