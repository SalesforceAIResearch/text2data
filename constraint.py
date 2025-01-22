import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from constraint.utils import utils
from collections import defaultdict

def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

class Constraint(torch.optim.Optimizer):
    """
    first_step: gradient of objective 1, and log the grad,
    second_step: gradient of objective 2, and do something based on the logged gradient at step one
    closure: the objective 2 for second step
    """

    def __init__(self, params, base_optimizer, alpha=1, beta=1, **kwargs):
        defaults = dict(**kwargs)
        super(Constraint, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.beta = beta
        self.g_constraint = 0.
        self.g_value = torch.tensor([1.]).item()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                constraint_grad = torch.ones_like(p.grad) * p.grad  # deepcopy, otherwise the c_grad would be a pointer
                self.state[p]["constraint_grad"] = constraint_grad

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        '''
        calculate the projection here
        '''
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                phi_x = min(self.alpha * (self.g_value - self.g_constraint), self.beta * torch.norm(self.state[p]["constraint_grad"]) ** 2)
                adaptive_step_x = F.relu((phi_x - (p.grad * self.state[p]["constraint_grad"]).sum()) / (1e-8 + self.state[p]["constraint_grad"].norm().pow(2)))
                p.grad.add_(adaptive_step_x * self.state[p]["constraint_grad"])

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, g_value=None, g_constraint=None):
        assert closure is not None, "Requires closure, but it was not provided, raise an error"
        assert g_value is not None, "Requires g value"
        assert g_constraint is not None, "Requires g constraint"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.g_value = g_value
        self.g_constraint = g_constraint
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def calculate_dr_grad_norm(self, model):
        total_norm = defaultdict(int)
        for idxx, p in enumerate(model.parameters()):
            param_norm = float(get_numpy(p.grad.detach().data.norm(2))) if p.grad is not None and p.requires_grad else 0.
            total_norm[idxx // 2] += param_norm

        return total_norm
