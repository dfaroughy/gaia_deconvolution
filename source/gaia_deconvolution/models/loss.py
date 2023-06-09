
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from gaia_deconvolution.data.transform import GaiaTransform

def calculate_loss(model, data, args, loss_func=None, reduction=torch.mean):
	if not loss_func: loss_func = args.loss
	loss = reduction(loss_func(model, data, args))
	return loss

def neglogprob_loss(model, batch, args):
	batch = batch[:, :args.dim]
	batch = batch.to(args.device)
	loss = - model.log_prob(batch)
	return loss

def deconv_loss(model, batch, args):
	d = args.dim
	n = args.num_mc
	batch = batch.repeat_interleave(n, 0)                   # ABC... -> AABBCC...
	x = batch[:, :d]										# x = (x1, x2)
	cov_flat = batch[:, d:] 								# cov = (c11, c12, c21, c22)  flat
	cov_matrix = torch.reshape(cov_flat, (-1, d, d))        # cov = ((c11, c12),(c21, c22))  reshaped
	eps = torch.randn_like(x)                               # sample eps ~ N(0,1) 
	eps = torch.reshape(eps,(-1, d, 1))                     # reshapes eps dim(2) -> 2x1 vector 
	x_noisy = x + torch.squeeze(torch.bmm(cov_matrix, eps)) # x + sigma * eps
	x_noisy = x_noisy.to(args.device)									
	loss = - torch.logsumexp(torch.reshape(model.log_prob(x_noisy),(-1, n)),dim=-1) 
	loss +=  torch.log(torch.tensor(1.0 if not n else n))
	return loss

# def deconv_loss(model, batch, args):
	
# 	d = args.dim
# 	n = args.num_mc

# 	batch = batch.repeat_interleave(n, 0)#                   # ABC... -> AABBCC...
# 	batch = GaiaTransform(batch)

# 	x = batch.xv	
# 													          # x = (x1, x2)
# 	cov_flat = batch.cov 									  # cov = (c11, c12, c21, c22)  flat
# 	cov_matrix = torch.reshape(cov_flat, (-1, d, d))#         # cov = ((c11, c12),(c21, c22))  reshaped
# 	eps = torch.randn_like(x)                                 # sample eps ~ N(0,1) 
# 	eps = torch.reshape(eps,(-1, d, 1))                       # reshapes eps dim(2) -> 2x1 vector 
# 	x_noisy = x + torch.squeeze(torch.bmm(cov_matrix, eps))            # x + sigma * eps
# 	loss = - torch.logsumexp(torch.reshape(model.log_prob(x_noisy),(-1, n)),dim=-1) 
# 	loss +=  torch.log(torch.tensor(1.0 if not n else n))
# 	return loss


def neglogprob_joint_loss(model, batch, args):
	d = args.dim
	n = args.num_mc
	x = batch[:, :d]													# x = (x1, x2)
	cov_flat = batch[:, d:]   											# (c11, c12, c21, c22)
	cov_flat = torch.cat((cov_flat[:, :1], cov_flat[:, -1:]), dim=1 )     # keep diagonal elements only: (c11, c22)
	x_covs = torch.cat((x, cov_flat), dim=1)                            # (x1, x2, c11, c22)   
	loss = - model.log_prob(x_covs)
	return loss


def deconv_joint_loss(model, batch, args):
	d = args.dim
	n = args.num_mc
	batch = batch.repeat_interleave(n,0)                               # ABC... -> AAABBBCCC...
	x = batch[:, :d]												   # x = (x1, x2)
	cov_flat = batch[:, d:]   									       # (c11, c12, c21, c22)
	cov_matrix = torch.reshape(cov_flat, (-1, d, d))                   # reshapes ((c11, c12),(c21, c22))
	cov_flat = torch.cat((cov_flat[:,:1], cov_flat[:,-1:]), dim=1 )    # keep diagonal elements only: (c11, c22)
	eps = torch.randn_like(x)                                         # sample eps ~ N(0,1) 
	eps = torch.reshape(eps,(-1, d, 1))                               # reshapes eps dim(2) -> 2x1 vector 
	x_noisy = x + torch.squeeze(torch.bmm(cov_matrix, eps))           # data + cov*eps
	x_covs_noisy = torch.cat((x_noisy, cov_flat), dim=1)              # (x1, x2, c11, c22)       
	loss = - torch.logsumexp(torch.reshape(model.log_prob(x_covs_noisy),(-1, n)),dim=-1)
	loss +=  torch.log(torch.tensor(1.0 if not n else n))
	return loss
