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
from gaia_deconvolution.models.training import GaiaModel
from gaia_deconvolution.models.loss import neglogprob_loss, deconv_loss
from gaia_deconvolution.data.transform import GaiaTransform

sys.path.append("../")
torch.set_default_dtype(torch.float64)


'''
    Description:

    Normalizing flow (maf or coupling layer) for denoising 
    the Gaia observation errors with a flow-based deconvolution. 
    The procedure consists of two steps:

        1. pretraining: first estimate the noisy phase-space density  
           with a normalizing flow (maf or coupling).learning can be crude.
           the loss is the negative log probability.

        2. deconvolution: use the pretrained model as input and train 
           over it using the deconvolution loss --loss = deconv_loss.
           the doconvolution loss performs a MC integration with number
           of samples controled by --num_mc. 
    
    tips:

        For the data to fit in GPU it is preffereble to sub-batch eahc batch 
        in steps (--num_steps) and perform gradient accumulation. 


'''


####################################################################################################################

params = argparse.ArgumentParser(description='arguments for the deconvolution model for p(w) -> p(v)')

params.add_argument('--workdir',      help='working directory', type=str)
params.add_argument('--device',       default='cuda:0',         help='where to train')
params.add_argument('--dim',          default=6,                help='dimensionalaty of data: (x,y,z,vx,vy,vz)', type=int)
params.add_argument('--num_mc',       default=200,             help='number of MC samples for integration', type=int)
params.add_argument('--loss',         default=deconv_loss,      help='loss function')
params.add_argument('--pretrain',     default=True,             help='if True, pretrain the flow on the noisy data before deconvoling', type=bool)

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

params.add_argument('--batch_size',      default=512,         help='size of training/testing batch', type=int)
params.add_argument('--num_steps',       default=10,         help='split batch into n_steps sub-batches + gradient accumulation', type=int)
params.add_argument('--test_size',       default=0.2,          help='fraction of testing data', type=float)
params.add_argument('--max_epochs',      default=100,           help='max num of training epochs', type=int)
params.add_argument('--max_patience',    default=10,           help='terminate if test loss is not changing', type=int)
params.add_argument('--lr',              default=1e-4,         help='learning rate of generator optimizer', type=float)
params.add_argument('--activation',      default=F.leaky_relu, help='activation function for neural networks')
params.add_argument('--batch_norm',      default=True,         help='apply batch normalization layer to flow blocks', type=bool)
params.add_argument('--dropout',         default=0.1,          help='dropout probability', type=float)
params.add_argument('--seed',            default=9999,          help='random seed for data split', type=int)

#... data params:

params.add_argument('--x_sun',      default=[8.122, 0.0, 0.0208],  help='sun position [kpc] wrt galactic center', type=list)
params.add_argument('--radius',     default=1.5,                   help='only keep stars within radius [kpc] of sun', type=float)
params.add_argument('--num_gen',    default=5000,                help='number of sampled stars from model', type=int)
params.add_argument('--num_stars',  default=0,                     help='total number of stars used for train/testing', type=int)
params.add_argument('--mean',       default=[],                    help='data mean (for preprocessing)', type=list)
params.add_argument('--std',        default=[],                    help='data covariance (for preprocessing)', type=list)
params.add_argument('--Rmax',       default=0.,                    help='maximum radius of smeared stars (for preprocessing)', type=float)


xlim = ((6,10),(-2,2),(-2,2))
ylim = ((-2,2),(-2,2),(6,10))


#... pretrain params: define new parser for the pre-training model (only necesary if --pretrain=True):

params_pre = copy_parser(params, 
                         description='arguments for the pre-trining model: flow learns the noisy distribution p(w)',
                         modifications={
                                        'loss' : {'default' : neglogprob_loss}, 
                                        'batch_size' : {'default' : 512},
                                        'num_steps' : {'default' : False}, 
                                        'num_mc' : {'default' : 0},
                                        'max_epochs' :  {'default' : 2},
                                        'max_patience' :  {'default' : 10} 
                                        } 
                        )

####################################################################################################################

if __name__ == '__main__':

    #...create working folders and save args

    args = params.parse_args()
    args.workdir = make_dir('Results_Gaia_Deconvolution', sub_dirs=['data_plots', 'results_plots'], overwrite=False)
    print("#================================================")
    print("INFO: working directory: {}".format(args.workdir))
    print("#================================================")

    #...get datasets

    data_file =  "./data/data.angle_340.smeared_00.npy"
    covs_file =  "./data/data.angle_340.smeared_00.cov.npy"
    data = torch.tensor(np.load(data_file))
    covs = torch.squeeze(torch.reshape( torch.tensor(np.load(covs_file)), (-1, 1, 6*6)))
    
    #...smear and preprocess data

    gaia = GaiaTransform(data, covs, args)
    gaia.get_stars_near_sun(R=args.radius)
    gaia.plot('x', title='target positions', save_dir=args.workdir+'/data_plots', xlim=xlim, ylim=ylim, bin_size=0.05) 
    gaia.plot('v', title='target velocities', save_dir=args.workdir+'/data_plots') 
    gaia.smear()
    gaia.preprocess()                        

    #...store parser args

    args.num_stars = gaia.num_stars
    args.mean = gaia.mean.tolist()
    args.std = gaia.std.tolist()
    args.Rmax = gaia.Rmax.tolist()
    print("INFO: num stars: {}".format(args.num_stars))
    save_arguments(args, name='inputs.json')

    #...define model

    if args.flow == 'MAF': flow = masked_autoregressive_flow(args)
    elif args.flow == 'coupling': flow = coupling_flow(args)
    flow = flow.to(args.device)    
    model = GaiaModel(flow)

    #...prepare train/test samples

    train, test  = train_test_split(gaia.data, test_size=args.test_size, random_state=args.seed)

    #...pretrain flow to estimate noisy phase-space

    if args.pretrain:

        print("INFO: start pre-training")
        args_pre = params_pre.parse_args()
        args_pre.workdir = args.workdir 
        args_pre.num_stars = args.num_stars
        args_pre.mean = args.mean
        args_pre.std = args.std
        args_pre.Rmax = args.Rmax 
        save_arguments(args_pre, name='input_pretrain.json')

        train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args_pre.batch_size, shuffle=True)
        test_sample  = DataLoader(dataset=torch.Tensor(test),  batch_size=args_pre.batch_size, shuffle=False)

        model.train(train_sample, test_sample, args_pre, show_plots=False, save_best_state=False)

        # sample from model and transform back to phase-space:

        sample = model.sample(num_samples=args.num_gen)
        gaia_sample = GaiaTransform(sample, torch.zeros(sample.shape), args_pre) 
        gaia_sample.mean = gaia.mean
        gaia_sample.std = gaia.std
        gaia_sample.preprocess(R=gaia.Rmax, reverse=True)
        gaia_sample.plot('x', title='pretrained position density', save_dir=args.workdir+'/results_plots', xlim=xlim, ylim=ylim, bin_size=0.05) 
        gaia_sample.plot('v', title='pretrained velocity density', save_dir=args.workdir+'/results_plots') 

    #... apply deconvolution on pretrained flow model

    print("INFO: start deconvolution")
    train_sample = DataLoader(dataset=torch.Tensor(train), batch_size=args.batch_size, shuffle=True)
    test_sample  = DataLoader(dataset=torch.Tensor(test),  batch_size=args.batch_size, shuffle=False)

    model.train(train_sample, test_sample, args)

    # sample from deconvoluted model and transform back to phase-space:

    sample = model.sample(num_samples=args.num_gen)
    gaia_sample_deconv = GaiaTransform(sample, torch.zeros(sample.shape), args) 
    gaia_sample_deconv.mean = gaia.mean
    gaia_sample_deconv.std =  gaia.std
    gaia_sample_deconv.preprocess(R=gaia.Rmax, reverse=True)
    gaia_sample_deconv.plot('x', title='deconvoluted position density', save_dir=args.workdir+'/results_plots', xlim=xlim, ylim=ylim, bin_size=0.05) 
    gaia_sample_deconv.plot('v', title='deconvoluted velocity density', save_dir=args.workdir+'/results_plots') 

