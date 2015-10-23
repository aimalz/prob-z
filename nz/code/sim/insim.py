"""
in-sim module contains parameters controlling one run of data generation
"""

# TO DO: add docstrings

import os
import shutil
import cPickle as cpkl
import timeit
import math as m
import numpy as np

# setup object contains inputs of necessary parameters for code
class setup(object):
    """
    setup object specifies all parameters controlling one run of p(z) inference program
    """
    def __init__(self,input_address):

        self.inadd = input_address[:-4]
        self.testdir = os.path.join('..','tests')
        self.datadir = os.path.join(self.testdir,input_address)

        # read input parameters
        with open(self.datadir) as infile:
            lines = (line.split(None) for line in infile)
            indict   = {defn[0]:defn[1:] for defn in lines}

        # take in specification of bins if provided, otherwise make some

        if 'allzs' in indict:
            self.allnbins = len(indict['allzs'])-1
            self.allzs = np.array([float(indict['allzs'][i]) for i in range(0,self.allnbins+1)])
        elif 'nbins' in indict:
            self.allnbins = int(indict['nbins'][0])
            binstep = 1. / self.allnbins
            self.allzs = np.arange(0.,1.+binstep,binstep)
        else:
            self.allnbins = 20
            binstep = 1. / self.allnbins
            self.allzs = np.arange(0.,1.+binstep,binstep)


        # his, los, centers of bins and bin widths handy for plotting
        self.allzlos = self.allzs[:-1]
        self.allzhis = self.allzs[1:]
        self.allzmids = (self.allzhis + self.allzlos) / 2
        self.zdifs = self.allzhis - self.allzlos
        self.zdif = sum(self.zdifs) / self.allnbins

        # define a physical P(z)
        # for sum of Gaussians, elements of the form (z_center, spread, magnitude)
        # take in specification of underlying P(z) if provided, otherwise make one
        zrange = max(self.allzs)-min(self.allzs)
        zmin = min(self.allzs)
        if 'phys' in indict:
            nelem = len(indict['phys'])
            self.real = np.array([float(indict['phys'][i]) for i in range(1,nelem)])
        else:
            self.real = np.array([[zmin+0.2*zrange, 0.005, 2.0],
                         [zmin+0.4*zrange, 0.005, 1.25],
                         [zmin+0.5*zrange, 0.1, 2.0],
                         [zmin+0.6*zrange, 0.005, 1.25],
                         [zmin+0.8*zrange, 0.005, 1.25],
                         [zmin+1.0*zrange, 0.005, 0.75]
                          ])
        # put together Gaussian elements
        self.realistic_comps = np.transpose([[zmid*tup[2]*(2*m.pi*tup[1])**-0.5*m.exp(-(zmid-tup[0])**2/(2*tup[1])) for zmid in self.allzmids] for tup in self.real])
        self.realistic = np.array([sum(realistic_comp) for realistic_comp in self.realistic_comps])

        # set up info for one test

        # dimensionality/ies of N(z) parameter
        if 'params' in indict:
            self.params = int(indict['params'][0])
            assert int(indict['params'][0]) < self.allnbins
        else:
            self.params = self.allnbins

        # generate number of galaxies in survey/s
        if 'survs' in indict:
            self.survs = int(indict['survs'][0])
        else:
            self.survs = 1000

        # 0 for set number of galaxies, 1 for statistical sample around target
        if 'poisson' in indict:
            self.poisson = bool(int(indict['poisson'][0]))
        else:
            self.poisson = bool(1)

        # assign zs to galaxies
        # 0 for all galaxies having same true redshift bin, 1 for statistical sample around underlying P(z)
        if 'random' in indict:
            self.random = bool(int(indict['random'][0]))
        else:
            self.random = bool(1)

        # 0 for all galaxies having mean redshift of bin, 1 for uniform sampling
        if 'uniform' in indict:
            self.uniform = bool(int(indict['uniform'][0]))
        else:
            self.uniform = bool(1)

        # permit more complicated p(z)s
        # 0 for unimodal, 1 for multimodal
        if 'shape' in indict:
            self.shape = bool(int(indict['shape'][0]))
        else:
            self.shape = bool(1)

        # 0 for noiseless, 1 for noisy
        if 'noise' in indict:
            self.noise = bool(int(indict['noise'][0]))
        else:
            self.noise = bool(1)

        # colors for plots
        self.colors='brgycm'

        self.topdir = os.path.join(self.testdir,self.inadd)#os.path.join(self.testdir,str(round(timeit.default_timer())))
        if os.path.exists(self.topdir):
            shutil.rmtree(self.topdir)
        os.makedirs(self.topdir)

#         topdir = os.path.join(self.testdir,'topdirs.p')
#         if os.path.isfile(topdir):
#               with open(os.path.join(self.testdir,'topdirs.p'),'rb') as topdir:
#                     topdirs = cpkl.load(topdir)#cPickle.dump(self.topdir,topdir)
#                     print('reading '+str(topdirs))
#               topdirs[input_address] = self.topdir
#               with open(os.path.join(self.testdir,'topdirs.p'),'wb') as topdir:
#                     cpkl.dump(topdirs,topdir)
#                     print('writing '+str(topdirs))
#         else:
#               topdirs = {input_address:self.topdir}
#               with open(os.path.join(self.testdir,'topdirs.p'),'wb') as topdir:
#                     cpkl.dump({input_address:self.topdir},topdir)
#                     print('writing '+str(topdirs))

        self.simdir = os.path.join(self.topdir,'data')
        if os.path.exists(self.simdir):
            shutil.rmtree(self.simdir)
        os.makedirs(self.simdir)

        outdict = {
            'topdir': self.topdir,
            'allzs': self.allzs,
            'params': self.params,
            'survs': self.survs,
            'poisson': self.poisson,
            'random': self.random,
            'uniform': self.uniform,
            'shape': self.shape,
            'noise': self.noise,
            }

        readme = open(os.path.join(self.simdir,'README.md'), 'w' )
        readme.write(repr(outdict))
        readme.close()

        print('ingested inputs')
# want to make data accessible later. . .
