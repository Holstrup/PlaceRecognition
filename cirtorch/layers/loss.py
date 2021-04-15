import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import cirtorch.layers.functional as LF

# --------------------------------------
# Loss/Error layers
# --------------------------------------

class ContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        return LF.contrastive_loss(x, label, margin=self.margin, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class LinearWeightedContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6, gpsmargin=15):
        super(LinearWeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.weighting = 0
        self.gpsmargin = gpsmargin

    def forward(self, x, label, gps=[]):
        loss, weighting = LF.linear_weighted_contrastive_loss(x, label, gps, margin=self.margin, eps=self.eps, gpsmargin=self.gpsmargin)
        self.weighting = weighting
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class LinearOverWeightedContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6, gpsmargin=15):
        super(LinearOverWeightedContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.weighting = 0
        self.gpsmargin = gpsmargin

    def forward(self, x, label, gps=[]):
        loss, weighting = LF.linear_over_weighted_contrastive_loss(x, label, gps, margin=self.margin, eps=self.eps, gpsmargin=self.gpsmargin)
        self.weighting = weighting
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class RegressionContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6, gpsmargin=15):
        super(RegressionContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.weighting = 0
        self.gpsmargin = gpsmargin
    
    def forward(self, x, label, gps=[]):
        loss, weighting = LF.regression_contrastive_loss(x, label, gps, margin=self.margin, eps=self.eps, gpsmargin=self.gpsmargin)
        self.weighting = weighting
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class LogTobitLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6, gpsmargin=15, sigma=2):
        super(LogTobitLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.gpsmargin = gpsmargin
        self.sigma = sigma
        self.weighting = 0
    
    def forward(self, x, label, gps=[]):
        loss, pos_weight = LF.log_tobit(x, label, gps, margin=self.margin, eps=self.eps, gpsmargin=self.gpsmargin, sigma=self.sigma)
        self.weighting = pos_weight
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class LearntLogTobitLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6, gpsmargin=15, sigma=5.0, scaling=100):
        super(LearntLogTobitLoss, self).__init__()
        self.scaling = 300 #max(300 - epoch, 100) #Parameter(torch.ones(1)*scaling) #scaling
        self.margin = margin
        self.eps = eps
        self.gpsmargin = gpsmargin
        self.sigma = sigma
        self.weighting = 0
    
    def forward(self, x, label, gps=[], epoch = 1):
        self.scaling = 1/max(300 - epoch, 0)
        loss, pos_weight = LF.log_tobit(x, label, gps, margin=self.margin, eps=self.eps, gpsmargin=self.gpsmargin, sigma=self.sigma, scaling=self.scaling)
        self.weighting = pos_weight
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class ContrastiveLossVariant(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)

    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7

    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7, eps=1e-6, gpsmargin=25, beta=1.0):
        super(ContrastiveLossVariant, self).__init__()
        self.margin = gpsmargin / margin
        self.gpsmargin = gpsmargin
        self.beta = beta
        self.eps = eps
        self.mse_loss = 0

    def forward(self, x, label, gps=[]):
        #loss = LF.contrastive_loss_mse_reference(x, label, margin=self.gpsmargin, eps=self.eps)
        #loss = LF.contrastive_loss_plus_mse(x, label, gps, eps=self.eps, margin=self.gpsmargin, alpha=self.margin, beta=self.beta)
        #loss, mse_loss = LF.contrastive_loss_mse(x, label, gps, eps=self.eps, margin=self.gpsmargin, alpha=self.margin, beta=self.beta)
        loss = LF.contrastive_loss_mse_smoothed(x, label, gps, margin=25, eps=1e-6, alpha=35, smoothing=0.1)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'

class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return LF.triplet_loss(x, label, margin=self.margin)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'
