import torch
import numpy as np
from torch.autograd import Variable
from kornia.color import ycbcr_to_rgb

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
class Architect(object):

    def __init__(self, fusion, detection, f_opt, h_opt, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.fuse = fusion
        self.detect = detection
        self.f_opt = f_opt
        self.h_opt = h_opt

    def _construct_model_from_theta(self, theta):
        model_new = self.fuse.new()
        model_dict = self.fuse.network.state_dict()

        params, offset = {}, 0
        for k, v in self.fuse.network.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.network.load_state_dict(model_dict)
        return model_new

    def _compute_unrolled_model(self, sample, eta):
        loss, _ = self.fuse.criterion_generator(sample=sample)
        theta = _concat(self.fuse.network.parameters()).data
        try:
            moment = _concat(self.f_opt.state[v]['momentum_buffer'] for v in self.fuse.network.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.fuse.network.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, sample_train, sample_val, eta):

        self.h_opt.zero_grad()
        self._backward_step_unrolled(sample_train, sample_val, eta)
        self.h_opt.step()

    def _backward_step_unrolled(self, sample_train, sample_val, eta):
        unrolled_model = self._compute_unrolled_model(sample_train, eta)
        unrolled_fus = unrolled_model.forward(ir=sample_val['ir'], vi=sample_val['vi'])
        unrolled_fus = torch.cat([unrolled_fus, sample_val['cbcr']], dim=1)
        unrolled_fus = ycbcr_to_rgb(unrolled_fus)
        unrolled_loss, _ = self.detect.criterion(
                    imgs=unrolled_fus,
                    targets=sample_val['labels'])
        unrolled_loss.backward()

        dalpha = []
        for v in unrolled_model.new_hyper():
            if v.grad is None:
                dalpha.append(torch.zeros_like(v))
            else:
                dalpha.append(v.grad)

        vector = [v.grad.data for v in unrolled_model.network.parameters()]
        implicit_grads = self._hessian_vector_product(vector, sample_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.fuse.new_hyper(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _hessian_vector_product(self, vector, sample_train, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.fuse.network.parameters(), vector):
            p.data.add_(R, v)
        loss, _ = self.fuse.criterion_generator(sample=sample_train)
        grads_p = torch.autograd.grad(loss, self.fuse.new_hyper())

        for p, v in zip(self.fuse.network.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss, _ = self.fuse.criterion_generator(sample=sample_train)
        grads_n = torch.autograd.grad(loss, self.fuse.new_hyper())

        for p, v in zip(self.fuse.network.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]