from typing import Literal, List, Tuple
import torch
import torch.backends.cudnn
from loss.loss import GradientLoss, PerceptualLoss, GradientLoss2
from kornia.losses import SSIMLoss, TotalVariation
from model.models import Ours
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Fuse:
    def __init__(self, config, mode: Literal['train', 'inference']):
        self.config = config
        self.mode = mode
        modules = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        network = Ours()

        modules.append(network)
        self.network = network
        self.num_loss = 15
        self.hypers = Variable(1.25e-2 * torch.ones(self.num_loss).cuda(), requires_grad=True)
        self._hypers_parameters = [self.hypers]

        l1_loss = nn.L1Loss()
        modules.append(l1_loss)
        self.l1_loss = l1_loss

        l2_loss = nn.MSELoss()
        modules.append(l2_loss)
        self.l2_loss = l2_loss

        tv_loss = TotalVariation()
        modules.append(tv_loss)
        self.tv_loss = tv_loss

        grad_loss = GradientLoss()
        modules.append(grad_loss)
        self.grad_loss = grad_loss
        
        singlegrad_loss = GradientLoss2()
        modules.append(singlegrad_loss)
        self.singlegrad_loss = singlegrad_loss

        perc_loss = PerceptualLoss()
        modules.append(perc_loss)
        self.perc_loss = perc_loss

        ssim_loss = SSIMLoss(5)
        modules.append(ssim_loss)
        self.ssim_loss = ssim_loss

        _ = [x.to(device) for x in modules]
        
    def save_ckpt(self) -> dict:
        ckpt = {'fuse': self.network}
        return ckpt

    def forward(self, ir, vi):
        self.network.train()
        fus = self.network(ir, vi)
        return fus

    @torch.no_grad()
    def eval(self, ir, vi):
        self.network.eval()
        fus = self.network(ir, vi)
        return fus

    def new(self):
        model_new = Fuse(self.config, mode='train')
        for x, y in zip(model_new.new_hyper(), self.new_hyper()):
            x.data.copy_(y.data)
        return model_new

    def new_hyper(self):
        return [self.hypers]

    def hyper_parameters(self):
        return [self.hypers]

    def criterion_generator(self, sample: Tuple, given_hypers=None):

        ir = sample['ir']
        vi = sample['vi']
        ir_v = sample['ir_v']
        vi_v = sample['vi_v']
        fus = self.forward(ir, vi)
        if not given_hypers:
            h = F.softmax(self.hypers, dim=-1)
        else:
            h = given_hypers

        l1_vi_vsm = self.l1_loss((fus * vi_v), (vi* vi_v))
        l1_ir_vsm =  self.l1_loss((fus * ir_v), (ir* ir_v))
        l1_vi = self.l1_loss(fus, vi)
        l1_ir =  self.l1_loss(fus, ir)
        l1_max = self.l1_loss(fus, torch.max(vi, ir))

        l2_vi_vsm = self.l2_loss((fus * vi_v), (vi* vi_v))
        l2_ir_vsm =  self.l2_loss((fus * ir_v), (ir* ir_v))
        l2_vi = self.l2_loss(fus, vi)
        l2_ir =  self.l2_loss(fus, ir)
        l2_max =  self.l2_loss(fus, torch.max(vi, ir))

        ssim_ir = self.ssim_loss(fus, ir).mean()
        ssim_vi = self.ssim_loss(fus, vi).mean()
        grad_ir = self.singlegrad_loss(fus, ir)
        grad_vi = self.singlegrad_loss(fus, vi)
        grad_max = self.grad_loss(fus, ir, vi)
        perc_ir = self.perc_loss(fus, ir).mean()
        perc_vi = self.perc_loss(fus, vi).mean()

        loss_list = [l1_vi_vsm, l1_ir_vsm, l1_vi, l1_ir, l1_max, l2_vi_vsm, l2_ir_vsm, l2_vi, l2_ir, l2_max, ssim_ir, ssim_vi, grad_ir, grad_vi, grad_max, perc_ir, perc_vi]
        losses = l2_vi_vsm * 0.5 + l2_ir_vsm * 0.5 + l1_vi_vsm * h[0] + l1_ir_vsm * h[1] +  l1_vi * h[2] + l1_ir * h[3] + l1_max * h[4] +  l2_vi * h[5] + l2_ir * h[6] + l2_max * h[7] + ssim_ir * h[8] + ssim_vi * h[9] +  grad_ir * h[10] + grad_vi * h[11] + grad_max * h[12] + perc_ir * h[13] * 0.1 + perc_vi * h[14] * 0.1

        return losses, loss_list

    def param_groups(self) -> Tuple[List, List, List]:

        group = [], [], []
        tmp = get_param_groups(self.network)
        for idx in range(3):
            group[idx].extend(tmp[idx])
        return group

def get_param_groups(module) -> Tuple[List, List, List]:
    group = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in module.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            "bias"
            group[2].append(v.bias)
        if isinstance(v, bn):
            "weight (no decay)"
            group[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            "weight (with decay)"
            group[0].append(v.weight)
    return group