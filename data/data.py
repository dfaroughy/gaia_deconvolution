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

from plotting import plot_data_projections


class GaiaTransform:

    def __init__(self, data, covs, args):
        
        self.args = args
        self.data = torch.cat((data, covs), dim=1)
        self.mean = torch.zeros(6)
        self.std = torch.zeros(6)

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
    def covs(self):
        return self.data[:,6:]

    @property
    def num_stars(self):
        return self.data.shape[0]

    def get_stars_near_sun(self, R=None):
        if not R: R=self.args.radius 
        distance = torch.norm(self.data[:, :3] - torch.tensor(self.args.x_sun), dim=-1)
        self.data = self.data[ distance < R] 
        return self

    def smear(self):
        covs_matrix = torch.reshape(self.covs, (-1, 6, 6)) 
        noise_dist = torch.distributions.MultivariateNormal(torch.zeros(self.num_stars, 6), covs_matrix)
        noise = noise_dist.sample()
        self.data[:,:6] = self.data[:,:6] + noise 
        return self

    def to_unit_ball(self, R, x0, inverse=False): 

        distances = torch.norm(self.data[:,:3] - x0, dim=1)
        R = torch.max(distances)

        if inverse: 
            self.data[:,:3] = (self.x * R ) + x0 
        else:  
            self.data[:,:3] = (self.x - x0) / R
        return self

    def radial_blowup_transform(self, inverse=False):
        x_norm = torch.linalg.norm(self.x, dim=-1, keepdims=True)
        if inverse: 
            self.data[:,:3] = (self.x / x_norm) * torch.tanh(x_norm)
        else: 
            self.data[:,:3] =  (self.x / x_norm)  * torch.atanh(x_norm)
        return self

    def standardization(self, inverse=False): 

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

    def preprocess(self, revert=False):  

        R = self.args.radius * (1+1e-6)
        x0 = torch.tensor(self.args.x_sun)

        if revert: 
            self.standardization(inverse=True)
            self.radial_blowup_transform(inverse=True)
            self.to_unit_ball(R=R, x0=x0, inverse=True)
        else: 
            self.to_unit_ball(R=R, x0=x0)
            self.radial_blowup_transform()
            self.standardization()

        return self 
