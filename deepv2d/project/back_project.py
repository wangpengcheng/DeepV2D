import math
from torch import nn
from torch.nn import Module, Parameter
from torch.autograd import Function
import torch
import numpy as np

from torch.autograd import Variable
import backproject

torch.manual_seed(42)


class BackProjectFunction(Function):
    @staticmethod
    def forward(ctx, inputs, coords):
        outputs = backproject.forward(
            inputs,
            coords
            )
       
        ctx.save_for_backward(inputs, coords)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, coords = ctx.saved_variables
        inputs_grad, coords_grad = backproject.backward(
            inputs, 
            coords,
            grad_output
            )
        return inputs_grad, coords_grad
        
class BackProject(Module):

    def __init__(self):
        super(BackProject, self).__init__()

    def forward(self, input, coords):
        return BackProjectFunction.apply(input, coords)
