import warnings

from torch.autograd import Variable
import torch
from torch.nn.modules.module import Module
from torch.nn.modules.container import Sequential
from torch.nn.modules.activation import LogSoftmax
from torch.nn import functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True, reduce=True):
        super(_WeightedLoss, self).__init__(size_average, reduce)
        self.register_buffer('weight', weight)


class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:
    The loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,
    where :math:`N` is the batch size. If reduce is ``True``, then:
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}
    `x` and `y` arbitrary shapes with a total of `n` elements each.
    The sum operation still operates over all the elements, and divides by `n`.
    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.
    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           for each minibatch. When reduce is ``False``, the loss function returns
           a loss per input/target element instead and ignores size_average.
           Default: ``True``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input
    Examples::
        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=True, reduce=True):
        super(L1Loss, self).__init__(size_average, reduce)

    def forward(self, input, target, mask):
        _assert_no_grad(target)
        return F.l1_loss(input * mask, target * mask, size_average=self.size_average,
                         reduce=self.reduce)