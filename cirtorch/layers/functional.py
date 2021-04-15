import math
import pdb

import torch
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt
import numpy as np

# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    _, idx = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b).int() - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b).int() - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2,i_,wl).narrow(3,j_,wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)


# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

def powerlaw(x, eps=1e-6):
    x = x + self.eps
    return x.abs().sqrt().mul(x.sign())

# --------------------------------------
# loss
# --------------------------------------

def contrastive_loss(x, label, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def regression_contrastive_loss(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    
    peak_scaling = 0
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
        peak_scaling = margin / gpsmargin * dist

    y = 4*lbl*torch.pow((D-peak_scaling),2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y, peak_scaling

def linear_weighted_contrastive_loss(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n
    
    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    
    weighting = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
        weighting = torch.max(1 - torch.div(dist,gpsmargin), torch.tensor([0.])) # Safety precaution. Should not happen.  
    weighting = weighting.cuda()
    y = 0.5*lbl*torch.pow(D,2)*weighting + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y, weighting

def linear_over_weighted_contrastive_loss(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n
    
    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    
    weighting = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
        weighting = max(2 - torch.div(2*dist, gpsmargin), 0) # Safety precaution. Should not happen.  
    
    y = 0.5*lbl*torch.pow(D,2)*weighting + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y, weighting

def logistically_weighted_contrastive_loss(x, label, gps, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    
    dist = distance(gps[0], gps[1])
    weighting = torch.div(1, 1 + torch.exp(dist - 15))
    
    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(y)
    return y

def distance(query, positive):
    return np.linalg.norm(np.array(query)-np.array(positive))

def triplet_loss(x, label, margin=0.1):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1).item() # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = x[:, label.data==0]

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))


def log_tobit_iteration1(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15, sigma=1/2):
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    dist = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
    scaling = 1/gpsmargin

    normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=torch.device("cuda")), torch.tensor([1.0]).to(device=torch.device("cuda")))
    log_tobit = - (math.log(1/sigma) + normal.log_prob((dist*scaling-D)/sigma)*lbl) - torch.log(normal.cdf((D - gpsmargin*scaling)/sigma)) * (1-lbl)
    y = torch.sum(log_tobit)
    return y, log_tobit[0]

def log_tobit_iteration2(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15, sigma=1):
    # Loss with only the CDF term 

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=torch.device("cuda")), torch.tensor([1.0]).to(device=torch.device("cuda")))
    scaling = 1 / (gpsmargin * 2)
    cdf = (normal.cdf((D - gpsmargin*scaling)/sigma))*lbl + (1-normal.cdf((D - gpsmargin*scaling)/sigma)) * (1-lbl)
    y = torch.sum(cdf)
    return y, cdf[0]

def log_tobit_iteration3(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15, sigma=1):
    # Loss with only the CDF term 

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=torch.device("cuda")), torch.tensor([1.0]).to(device=torch.device("cuda")))
    scaling = 1 / (gpsmargin * 2)
    cdf = -torch.log(1-normal.cdf((D - gpsmargin*scaling)/sigma))*lbl - torch.log(normal.cdf((D - gpsmargin*scaling)/sigma)) * (1-lbl)
    y = torch.sum(cdf)
    return y, cdf[0]


def log_tobit_iteration4(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15, sigma=5.0, scaling=100, beta=0):
    # CDF reverse engineered to look like contrastive 

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    dist = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
    
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=torch.device("cuda")), torch.tensor([sigma]).to(device=torch.device("cuda")))
    cdf = (-torch.log(1-normal.cdf((D*scaling - dist)/sigma))*beta + 0.5*lbl*torch.pow(D,2)*(1-beta))*lbl - torch.log(normal.cdf((D*scaling - gpsmargin)/sigma)) * (1-lbl)
    y = torch.sum(cdf)
    return y, D[0]*scaling

def log_tobit_iteration5(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15, sigma=5.0, scaling=100):
    # CDF reverse engineered to look like contrastive 

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    dist = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
    
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=torch.device("cuda")), torch.tensor([sigma]).to(device=torch.device("cuda")))
    
    # Tobit Positive * lbl + Contrastive Negative * (1-lbl)
    cdf = lbl*(math.log(1/sigma) + normal.log_prob((dist - D*scaling) / sigma)) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
    y = torch.sum(cdf)
    return y, D[0]*scaling

def log_tobit(x, label, gps, margin=0.7, eps=1e-6, gpsmargin=15, sigma=1.0, scaling=100):
    # CDF reverse engineered to look like contrastive 

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()

    dist = 1
    if len(gps) > 0:
        dist = distance(gps[0], gps[1])
    
    normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=torch.device("cuda")), torch.tensor([sigma]).to(device=torch.device("cuda")))
    cdf = -lbl*(math.log(1/sigma) + normal.log_prob((dist*scaling - D) / sigma)) - torch.log(normal.cdf((D - gpsmargin*scaling)/sigma)) * (1-lbl)
    y = torch.sum(cdf)
    return y, D[0]

def contrastive_loss_mse_reference(x, label, margin=25, eps=1e-6): 
    # Regular contrastive loss scaled up and then down again (sanity check)
    
    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt() 
    alpha = torch.ones(lbl.size()) * 35
    alpha = alpha.cuda()
    D = D * alpha 
    y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)    

    y /= (35**2)
    y = torch.sum(y)
    return y

def contrastive_loss_plus_mse(x, label, gps, margin=25, eps=1e-6, alpha=35, beta=0.5): 
    # Regular contrastive loss scaled up and then down again (sanity check)

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    alpha = torch.ones(lbl.size()) * alpha
    alpha = alpha.cuda()    
    D = D * alpha 

    gps_dist = 1
    if len(gps) > 0:
        gps_dist = distance(gps[0], gps[1])
 
    y = (1-beta)*0.5*lbl*torch.pow(D,2) + lbl * beta * torch.pow(D - gps_dist, 2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)  

    y /= (alpha**2)
    y = torch.sum(y)
    return y

def contrastive_loss_mse(x, label, gps, margin=25, eps=1e-6, alpha=35): 
    # Regular contrastive loss scaled up and then down again (sanity check)

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    alpha = torch.ones(lbl.size()) * alpha
    alpha = alpha.cuda()    
    D = D * alpha 

    gps_dist = 1
    if len(gps) > 0:
        gps_dist = distance(gps[0], gps[1])

    y = lbl * torch.pow(D - gps_dist, 2) 
    mse_loss = y / (alpha**2)
    y += 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)  

    y /= (alpha**2)
    y = torch.sum(y)
    mse_loss = torch.sum(mse_loss)
    return y, mse_loss

def contrastive_loss_mse_smoothed(x, label, gps, margin=25, eps=1e-6, alpha=35, smoothing=0.1): 
    # Regular contrastive loss scaled up and then down again (sanity check)

    # x is D x N
    dim = x.size(0) # D
    nq = torch.sum(label.data==-1) # number of tuples
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label!=-1]

    dif = x1 - x2
    D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
    alpha = torch.ones(lbl.size()) * alpha
    alpha = alpha.cuda()    
    D = D * alpha 

    smoothing = torch.zeros(len(lbl))
    for i, gps_i in enumerate(gps[1:]):
        smoothing[i] = 1 - distance(gps[0], gps_i)/50
    smoothing = smoothing.cuda()

    y = 0.5*lbl*smoothing*torch.pow(D,2) + 0.5*(1-lbl)*smoothing*torch.pow(torch.clamp(margin-D, min=0),2)

    y /= (alpha**2)
    y = torch.sum(y)
    mse_loss = torch.sum(mse_loss)
    return y