import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import sys
import copy
from copy import deepcopy
import argparse
import json

from utils import *
from data import GaiaTransform
from plotting import plot_data_projections

from flow_models import Normalizing_Flow, sampler
from training import Train_Model
from loss_functions import neglogprob_loss, deconv_loss

sys.path.append("../")
torch.set_default_dtype(torch.float64)

####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the deconvolution model')

params.add_argument('--workdir',      help='working directory', type=str)
params.add_argument('--device',       default='cuda:1',         help='where to train')
params.add_argument('--dim',          default=6,                help='dimensionalaty of data: (x,y,x,vx,vy,vz)', type=int)
params.add_argument('--num_mc',       default=100,              help='number of MC samples for integration', type=int)
params.add_argument('--loss',         default=deconv_loss,      help='loss function')
params.add_argument('--pretrain',     default=True,             help='if True, pretrain the flow on the noisy data before deconvoling', type=bool)

#...flow params:

params.add_argument('--flow_dim',     default=6,            help='dimensionalaty of input features for flow, usually same as --dim', type=int)
params.add_argument('--flow_type',    default='MAF',        help='type of flow model: coupling or MAF', type=str)
params.add_argument('--flow_func',    default='RQSpline',   help='type of flow transformation: affine or RQSpline', type=str)
params.add_argument('--coupl_mask',   default='checkers',   help='mask type: mid-split or checkers', type=str)
params.add_argument('--permutation',  default='inverse',    help='type of fixed permutation between flows: n-cycle or inverse', type=str)
params.add_argument('--num_flows',    default=5,            help='num of flow layers', type=int)
params.add_argument('--hidden_dims',  default=128,           help='dimension of hidden layers', type=int)
params.add_argument('--spline',       default=20,           help='num of spline for rational_quadratic', type=int)
params.add_argument('--num_blocks',   default=2,            help='num of MADE blocks in flow', type=int)
params.add_argument('--context_dim',  default=None,         help='dimension of context features', type=int)

#...training params:

params.add_argument('--lr',           default=1e-4,           help='learning rate of generator optimizer', type=float)
params.add_argument('--batch_size',   default=512,            help='size of training/testing batch', type=int)
params.add_argument('--batch_steps',  default=0,              help='set the number of sub-batch steps for gradient accumulation', type=int)
params.add_argument('--test_size',    default=0.2,            help='fraction of testing data', type=float)
params.add_argument('--activation',   default=F.leaky_relu,   help='activation function for neuarl networks')
params.add_argument('--max_epochs',   default=200,            help='max num of training epochs', type=int)
params.add_argument('--max_patience', default=20,             help='terminate if test loss is not changing', type=int)
params.add_argument('--batch_norm',   default=True,           help='apply batch normalization layer to flow blocks', type=bool)
params.add_argument('--dropout',      default=0.1,           help='dropout probability', type=float)

#... data params:

params.add_argument('--x_sun',      default=[8.122, 
                                             0.0, 
                                             0.0208],      help='sun position [kpc] wrt galactic center', type=list)

params.add_argument('--radius',     default=3.5,           help='only keep stars within radius [kpc] of sun', type=float)
params.add_argument('--num_stars',  default=None,          help='total number of stars used for train/testing', type=float)
params.add_argument('--num_gen',    default=50000,         help='number of sampled stars from model', type=int)
params.add_argument('--scale_cov',  default=10.0,          help='rescales the covariance matrix of Gaia noise data', type=float)

#... plot params:

xlim = [(3, 13), (-5, 5), (-5, 5)]
ylim = [(-5, 5), (-5, 5), (3, 13)]
vxlim = [(-400, 400), (-400, 400), (-400, 400)]
vylim = [(-400, 400), (-400, 400), (-400, 400)]
vlabel = [r'$v_x$ (kpc)',r'$v_y$ (kpc)',r'$v_z$ (kpc)']

#... pretrain params: define new parser for the pre-training model (only necesary if --pretrain=True):

params_pre = copy_parser(params, 
                         description='arguments for the pre-trining model: this flow learns the noisy data distribution before deconvoling',
                         modifications={'loss' : {'default' : neglogprob_loss}, 
                                        'batch_steps' : {'default' : False}, 
                                        'num_mc' : {'default' : 0},
                                        'max_epochs' :  {'default' : 50},
                                        'max_patience' :  {'default' : 10}
                                        } )

####################################################################################################################

if __name__ == '__main__':

    #...create working folders and save args

    args = params.parse_args()
    args.workdir = make_dir('Gaia_deconv_MC_{}'.format(args.num_mc), sub_dirs=['data', 'results'])
    print("#================================================")
    print("INFO: working directory: {}".format(args.workdir))
    print("#================================================")


    #...get datasets

    data_file =  "./data/data.angle_340.smeared_00.npy"
    covs_file=  "./data/data.angle_340.smeared_00.cov.npy"
    data = torch.tensor(np.load(data_file))
    covs = args.scale_cov * storch.squeeze(torch.reshape( torch.tensor(np.load(covs_file)), (-1, 1, 6*6)))
    
    gaia = GaiaTransform(data, covs, args)
    
    plot_data_projections(gaia.x, bin_size=0.1, num_stars=args.num_gen, xlim=xlim, ylim=ylim,  title=r'truth positions', save=args.workdir+'/data/truth_x.pdf')
    plot_data_projections(gaia.v, bin_size=5, num_stars=args.num_gen, xlim=vxlim, ylim=vylim,  label=vlabel, title=r'truth velocities', save=args.workdir+'/data/truth_v.pdf')
    
    #...smear data x7 times 

    gaia.smear()

    #...get all stars within 3.5 kpc from sun 

    gaia.get_stars_near_sun()   

    plot_data_projections(gaia.x, bin_size=0.1, num_stars=args.num_gen, xlim=xlim, ylim=ylim, title=r'smeared positions', save=args.workdir+'/data/smeared_x.pdf')
    plot_data_projections(gaia.v, bin_size=5, num_stars=args.num_gen, xlim=vxlim, ylim=vylim, label=vlabel, title=r'smeared velocities', save=args.workdir+'/data/smeared_v.pdf')
    
    #...apply preprocessing of data from Sung paper

    gaia.preprocess()
    
    plot_data_projections(gaia.x, bin_size=0.1, num_stars=args.num_gen, title=r'preprocessed smeared positions', save=args.workdir + '/data/preproc_smeared_x_.pdf')    
    plot_data_projections(gaia.v, bin_size=0.1, num_stars=args.num_gen, label=vlabel, title=r'preprocessed smeared velocities', save=args.workdir + '/data/preproc_smeared_v.pdf')                                  

    args.num_stars = gaia.num_stars
    print("INFO: num stars: {}".format(args.num_stars))
    save_arguments(args, name='inputs.json')

    #...Prepare train/test samples

    train, test = train_test_split(gaia.data, test_size=args.test_size, random_state=12385)
    train_sample = DataLoader(dataset=torch.Tensor(train).to(args.device),  
                              batch_size=args.batch_size, 
                              shuffle=True)
    test_sample = DataLoader(dataset=torch.Tensor(test).to(args.device), 
                             batch_size=args.batch_size,
                             shuffle=False)

    #...define model

    flow = Normalizing_Flow(args)

    if args.pretrain:

        print("INFO: pretrain flow: {}".format(args.pretrain))
        args_pre = params_pre.parse_args()
        args_pre.workdir = args.workdir 
        save_arguments(args_pre, name='input_pretrain.json')

        #...pretrain the flow to fit the noisy data before deconvolution

        flow = Train_Model(flow, train_sample, test_sample, args_pre , show_plots=False, save_best_state=False)
        sample = sampler(flow, num_samples=args.num_gen)
        gaia_sample = GaiaTransform(sample, torch.zeros(sample.shape), args) 
        gaia_sample.mean = gaia.mean
        gaia_sample.std =  gaia.std
        gaia_sample.preprocess(revert=True)
        plot_data_projections(gaia_sample.x, bin_size=0.1, num_stars=args.num_gen, xlim=xlim, ylim=ylim, title=r'pretrained noisy positions', save=args.workdir + '/results/pretrained_x_model.pdf')    
        plot_data_projections(gaia_sample.v, bin_size=5, num_stars=args.num_gen, xlim=vxlim, ylim=vylim, label=vlabel, title=r'pretrained noisy velocities', save=args.workdir + '/results/pretrained_v_model.pdf')                                  

    #...deconvolution

    deconvoluted = Train_Model(flow, train_sample, test_sample, args, show_plots=False, save_best_state=False)
    sample = sampler(deconvoluted, num_samples=args.num_gen)
    gaia_sample_deconv = GaiaTransform(sample, torch.zeros(sample.shape), args) 
    gaia_sample_deconv.mean = gaia.mean
    gaia_sample_deconv.std =  gaia.std
    gaia_sample_deconv.preprocess(revert=True)
    plot_data_projections(gaia_sample_deconv.x, bin_size=0.1, num_stars=args.num_gen, xlim=xlim, ylim=ylim, title=r'deconvoluted noisy positions', save=args.workdir + '/results/best_x_model.pdf')    
    plot_data_projections(gaia_sample_deconv.v, bin_size=5, num_stars=args.num_gen, xlim=vxlim, ylim=vylim, label=vlabel, title=r'deconvoluted noisy velocities', save=args.workdir + '/results/best_v_model.pdf')                                  


