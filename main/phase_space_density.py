import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import copy
from copy import deepcopy
import argparse
import json

from gaia_deconvolution.utils.base import make_dir, copy_parser, save_arguments
from gaia_deconvolution.models.flows.norm_flows import masked_autoregressive_flow, coupling_flow
from gaia_deconvolution.models.training import Train_Model, sampler
from gaia_deconvolution.models.loss import neglogprob_loss
from gaia_deconvolution.data.plots import plot_data_projections
from gaia_deconvolution.data.transform import GaiaTransform

sys.path.append("../")
torch.set_default_dtype(torch.float64)


'''
    Description:

    Normalizing flow (maf or coupling layer) for learning the 
    phase-space density of stars within 3.5 kpc from the sun. No deconvolution.

    tips:

     --data 'noisy': flow learns the noisy star density (fuzzy ball in position space)
     --data 'truth': flow learns the truth star density (hard-edged ball in position space)

'''

####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the flow model for phase-space density estimation')

params.add_argument('--workdir')
params.add_argument('--device',       default='cuda:1',           help='where to train')
params.add_argument('--dim',          default=6,                  help='dimensionalaty of data: (x,y,z,vx,vy,vz)', type=int)
params.add_argument('--loss',         default=neglogprob_loss,    help='loss function')

#...flow params:

params.add_argument('--flow',         default='coupling',   help='type of flow model: coupling or MAF', type=str)
params.add_argument('--dim_flow',     default=6,            help='dimensionalaty of input features for flow, usually same as --dim', type=int)
params.add_argument('--flow_func',    default='RQSpline',   help='type of flow transformation: affine or RQSpline', type=str)
params.add_argument('--coupl_mask',   default='mid-split',  help='mask type [only for coupling flows]: mid-split or checkers', type=str)
params.add_argument('--permutation',  default='inverse',    help='type of fixed permutation between flows: n-cycle or inverse', type=str)
params.add_argument('--num_flows',    default=10,            help='num of flow layers', type=int)
params.add_argument('--dim_hidden',   default=128,          help='dimension of hidden layers', type=int)
params.add_argument('--num_spline',   default=30,           help='num of spline for rational_quadratic', type=int)
params.add_argument('--num_blocks',   default=2,            help='num of MADE blocks in flow', type=int)
params.add_argument('--dim_context',  default=None,         help='dimension of context features', type=int)

#...training params:

params.add_argument('--batch_size',    default=1024,          help='size of training/testing batch', type=int)
params.add_argument('--num_steps',     default=0,            help='split batch into n_steps sub-batches + gradient accumulation', type=int)
params.add_argument('--test_size',     default=0.2,          help='fraction of testing data', type=float)
params.add_argument('--max_epochs',    default=1000,         help='max num of training epochs', type=int)
params.add_argument('--max_patience',  default=20,           help='terminate if test loss is not changing', type=int)
params.add_argument('--lr',            default=1e-4,         help='learning rate of generator optimizer', type=float)
params.add_argument('--activation',    default=F.leaky_relu, help='activation function for neural networks')
params.add_argument('--batch_norm',    default=True,         help='apply batch normalization layer to flow blocks', type=bool)
params.add_argument('--dropout',       default=0.1,          help='dropout probability', type=float)

#... data params:

params.add_argument('--data',       default='noisy',               help='noisy or truth data', type=str)
params.add_argument('--x_sun',      default=[8.122, 0.0, 0.0208],  help='sun position [kpc] wrt galactic center', type=list)
params.add_argument('--radius',     default=3.5,                   help='only keep stars within radius [kpc] of sun', type=float)
params.add_argument('--num_gen',    default=100000,                help='number of sampled stars from model', type=int)
params.add_argument('--num_stars',  default=0,                     help='total number of stars used for train/testing', type=int)
params.add_argument('--mean',       default=[],                    help='data mean (for preprocessing)', type=list)
params.add_argument('--std',        default=[],                    help='data covariance (for preprocessing)', type=list)
params.add_argument('--Rmax',       default=0.,                    help='maximum radius of smeared stars (for preprocessing)', type=float)


####################################################################################################################

if __name__ == '__main__':

    #...create working folders and save args

    args = params.parse_args()
    args.workdir = make_dir('Results_Gaia_phase-space_{}_density'.format(args.data), sub_dirs=['data_plots', 'results_plots'], overwrite=True)
    print("#================================================")
    print("INFO: working directory: {}".format(args.workdir))
    print("#================================================")

    #...get datasets, preprocess them

    data_file =  "./data/data.angle_340.smeared_00.npy"
    covs_file =  "./data/data.angle_340.smeared_00.cov.npy"
    data = torch.tensor(np.load(data_file))
    covs = torch.squeeze(torch.reshape( torch.tensor(np.load(covs_file)), (-1, 1, 6*6)))   

    #...smear and preprocess data

    gaia = GaiaTransform(data, covs, args)
    gaia.get_stars_near_sun(self, R=args.radius)

    if args.data == 'noisy': gaia.smear()

    gaia.plot('x', title='target positions', save_dir=args.workdir+'/data_plots') 
    gaia.plot('v', title='target velocities', save_dir=args.workdir+'/data_plots') 
    
    gaia.preprocess()

    #...store parser arguments

    args.num_stars = gaia.num_stars
    args.mean = gaia.mean.tolist()
    args.std = gaia.std.tolist()
    args.Rmax = gaia.Rmax.tolist()
    print("INFO: num stars: {}".format(args.num_stars))
    save_arguments(args, name='inputs.json')

    #...Prepare train/test samples

    train, test  = train_test_split(gaia.data, test_size=args.test_size, random_state=9999)
    train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args.batch_size, shuffle=True)
    test_sample  = DataLoader(dataset=torch.Tensor(test),  batch_size=args.batch_size, shuffle=False)

    #...define model

    if args.flow == 'MAF': flow = masked_autoregressive_flow(args)
    elif args.flow == 'coupling': flow = coupling_flow(args)

    flow = flow.to(args.device)

    #...train flow for phase-space density estimation.

    Train_Model(flow, train_sample, test_sample, args , show_plots=False, save_best_state=False)
    
    #...sample from flow model

    sample = sampler(flow, num_samples=args.num_gen)

    #...transofrm back to phase-space amd plot

    gaia_sample = GaiaTransform(sample, torch.zeros(sample.shape), args) 
    gaia_sample.mean = gaia.mean
    gaia_sample.std =  gaia.std
    gaia_sample.preprocess(R=gaia.Rmax, reverse=True) # invert preprocess transformations
    gaia_sample.plot('x', title='position density', save_dir=args.workdir+'/results_plots') 
    gaia_sample.plot('v', title='velocity density', save_dir=args.workdir+'/results_plots') 

    