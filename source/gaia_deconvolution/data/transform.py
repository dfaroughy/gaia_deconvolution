import time
import numpy as np
import pandas as pd
import torch
import astropy
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactocentric
from pygaia.errors import astrometric
from pygaia.errors import spectroscopic
from gaia_deconvolution.data.plots import plot_data_projections

class GaiaTransform:

    def __init__(self, data, args, covs=None):

        self.args = args
        if not covs: self.data = data
        else:  self.data = torch.cat((data, covs), dim=1)        
        self.mean = torch.zeros(6)
        self.std = torch.zeros(6)
        self.R = None

    @property
    def x(self):
        return self.data[:, :3]
    @property
    def v(self):
        return self.data[:, 3:6]
    @property
    def xv(self):
        return self.data[:, :6]
    @property
    def num_stars(self):
        return self.data.shape[0]
    @property
    def covs(self):
        return self.data[:,6:]

    def get_stars_near_sun(self, R=None, verbose=True):
        if not R: R=self.args.radius 
        if verbose: print('INFO: fetching stars within radius {} kpc from sun'.format(R))
        distance = torch.norm(self.data[:, :3] - torch.tensor(self.args.x_sun), dim=-1)
        self.data = self.data[ distance < R] 
        return self

    def smear(self, verbose=True):
        if verbose: print('INFO: smearing data')
        covs_matrix = torch.reshape(self.covs, (-1, 6, 6)) 
        noise_dist = torch.distributions.MultivariateNormal(torch.zeros(self.num_stars, 6), covs_matrix)
        noise = noise_dist.sample()
        self.data[:,:6] = self.data[:,:6] + noise 
        return self

    def to_unit_ball(self, R=None, inverse=False, verbose=True): 
        x0 = torch.tensor(self.args.x_sun)
        if R:
            self.R = R
        else:
            dist = torch.norm(self.data[:,:3] - x0, dim=-1)
            self.R = torch.max(dist) * (1+1e-6)
        if verbose: print('INFO: centering and scaling to unit ball at origin, scale={}'.format(R))
        if inverse: 
            self.data[:,:3] = (self.x * self.R ) + x0 
        else:  
            self.data[:,:3] = (self.x - x0) / self.R
        return self

    def radial_blowup_transform(self, inverse=False, verbose=True):
        if verbose: print('INFO: transform hard edge of data to infinity')
        x_norm = torch.linalg.norm(self.x, dim=-1, keepdims=True)
        if inverse: 
            self.data[:,:3] = (self.x / x_norm) * torch.tanh(x_norm)
        else: 
            self.data[:,:3] =  (self.x / x_norm)  * torch.atanh(x_norm)
        return self

    def standardization(self, inverse=False, verbose=True):
        if verbose: print('INFO: standardizing data') 
        if inverse: 
            self.data[:,:3] = self.x * self.std[:3] + self.mean[:3]
            self.data[:,3:6] = self.v * self.std[3:] + self.mean[3:]
        else: 
            self.mean[:3] = torch.mean(self.data[:,:3], dim=0)
            self.mean[3:] = torch.mean(self.data[:,3:6], dim=0)
            self.std[:3] = torch.std(self.data[:,:3], dim=0)
            self.std[3:] = torch.std(self.data[:,3:6], dim=0)
            self.data[:,:3] = (self.x - self.mean[:3]) / self.std[:3]
            self.data[:,3:6] = (self.v - self.mean[3:]) / self.std[3:]
        return self 

    def preprocess(self, R=None, reverse=False, verbose=True):  
        if verbose: print('INFO: preprocessing data')
        x0 = torch.tensor(self.args.x_sun)
        if R:
            self.R = R
        else:
            dist = torch.norm(self.data[:,:3] - x0, dim=-1)
            self.R = torch.max(dist) * (1+1e-6)
        if reverse: 
            self.standardization(inverse=True, verbose=False)
            self.radial_blowup_transform(inverse=True, verbose=False)
            self.to_unit_ball(R=R, inverse=True, verbose=False)
        else: 
            self.to_unit_ball(R=R, verbose=False)
            self.radial_blowup_transform( verbose=False)
            self.standardization(verbose=False)
        return self

    def plot(self, dat_type, title, 
             bin_size=0.1, 
             num_stars=None, 
             cmap="magma",
             xlim=None, 
             ylim=None, 
             save_dir=None):
        
        if isinstance(dat_type, str):  
            label = data_type
            if not xlim:
                if dat_type == 'x':
                    bin_size = 0.1
                    xlim = [(3, 13), (-5, 5), (-5, 5)]
                    ylim = [(-5, 5), (-5, 5), (3, 13)]
                if dat_type == 'v': 
                    bin_size = 5
                    xlim = [(-400, 400), (-400, 400), (-400, 400)]
                    ylim = [(-400, 400), (-400, 400), (-400, 400)]
            data = getattr(self, dat_type)

        if not num_stars: num_stars = self.args.num_gen
        if not save_dir: save_dir = self.args.workdir

        plot_data_projections(data, bin_size=bin_size, 
                                    num_stars=num_stars,
                                    label=label, 
                                    cmap=cmap, 
                                    xlim=xlim, 
                                    ylim=ylim, 
                                    title=title, 
                                    save_dir=save_dir)


    