"""
inputs module contains parameters controlling one run of p(z) inference program
"""

# TO DO: add docstrings
# TO DO: add option to generate data only, run only, plot only

import os
import sys
import cPickle
import timeit
import pyfits
import math as m
import numpy as np

# setup object contains inputs of necessary parameters for code
class setup(object):
    """
    setup object specifies all parameters controlling one run of p(z) inference program
    """
    def __init__(self):

        # Sheldon, et al. 2011 redshift bins
        #loc = "http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits"
        #zbins = pyfits.open(loc)
        #self.allnbins = len(zbins[1].data)
        #self.allzlos = np.array([zbins[1].data[k][0] for k in lrange(self.allnbins)])
        #self.allzhis = np.array([zbins[1].data[k][1] for k in lrange(self.allnbins)])
        #self.allzs = np.unique(np.concatenate([self.allzlos,self.allzhis]))

        # synthetic redshift bins
        self.allnbins = 10
        self.binstep = 1. / self.allnbins
        self.allzs = np.arange(0., 1. + self.binstep, self.binstep)
        self.allzlos = np.arange(0., 1., self.binstep)#self.allzs[:-1]
        self.allzhis = np.arange(self.binstep, 1. + self.binstep, self.binstep)#self.allzs[1:]

        # centers of bins and bin widths handy for plotting
        self.allzmids = (self.allzhis + self.allzlos) / 2
        self.zdifs = self.allzhis - self.allzlos
        self.zdif = sum(self.zdifs) / self.allnbins

        # define a physical P(z)
        # for sum of Gaussians, tuples of the form (z_center, spread, magnitude)
        self.real = [(0.2, 0.005, 2.0),
                     (0.4, 0.005, 1.25),
                     (0.5, 0.1, 2.0),
                     (0.6, 0.005, 1.25),
                     (0.8, 0.005, 1.25),
                     (1.0, 0.005, 0.75)]
        self.realistic_comps = np.transpose([[zmid*tup[2]*(2*m.pi*tup[1])**-0.5*m.exp(-(zmid-tup[0])**2/(2*tup[1])) for zmid in self.allzmids] for tup in self.real])
        self.realistic = np.array([sum(realistic_comp) for realistic_comp in self.realistic_comps])

        # set up info for all tests
        # should make this input directly from user

        # dimensionality/ies of N(z) parameter
        self.params = [self.allnbins]

        # generate number of galaxies in survey/s
        self.survs = [100]

        # instantiations of the survey (more than 1 breaks some plots)
        self.samps = 4
        self.poisson = [0,0,0,0]#0 for set number of galaxies, 1 for statistical sample around target
        self.random = [1,1,1,1]#0 for all galaxies having same true redshift bin, 1 for statistical sample around underlying P(z)
        self.uniform = [1,1,1,1]#0 for all galaxies having mean redshift of bin, 1 for uniform sampling

        # permit more complicated p(z)s
        self.shape = [0,0,1,1]#0 for unimodal, 1 for multimodal
        self.noise = [0,1,0,1]#0 for noiseless, 1 for noisy

        # initialization schemes
        self.init_names = ['Gaussian Ball Around Prior Sample']#may also include 'Prior Samples', 'Gaussian Ball Around Mean'
        self.inits = ['gs']#corresponding to 'ps', 'gm'

        # parameters for MCMC
        self.miniters = int(1e3)
        self.thinto = int(1e2)
        self.ntimes = self.miniters / self.thinto

        # colors for plots
        self.colors='rgbymc'

#         # save inputs to check with later, if identical don't regenerate data
#         self.olddata = 0
#         if os.path.isfile('test.p'):
#             lastrun = cPickle.load(open('test.p','rb'))
#             if lastrun == self:
#                 self.olddata = 1
#         else:
#             cPickle.dump(self,open('test.p','wb'))

        readme_dict = {
#             'topdir': self.topdir,
            'allzs': self.allzs,
            'params': self.params,
            'survs': self.survs,
            'samps': self.samps,
            'poisson': self.poisson,
            'random': self.random,
            'uniform': self.uniform,
            'shape': self.shape,
            'noise': self.noise,
            'inits': self.inits,
            'miniters': self.miniters,
            'thinto': self.thinto
            }

                # generate data y/n
        self.gendat = 1

        # do MCMC y/n
        self.mcmc = 1

        # make plots y/n
        self.plots = 1

        # topdir.p is there for debugging purposes, so I don't regenerate data if the problem happens after data is generated
        if (self.gendat == 0 or self.mcmc == 0 or self.plots == 0):
            assert os.path.isfile('topdir.p')
            self.topdir = cPickle.load(open('topdir.p','rb'))
            readme = open(os.path.join(self.topdir,'README.md'), 'r' ).read()
            assert repr(readme_dict) == readme
            print('loaded old run '+self.topdir)
        else:
            self.topdir = 'test'+str(round(timeit.default_timer()))
            cPickle.dump(self.topdir,open('topdir.p','wb'))
            os.makedirs(self.topdir)

        # make files to put progress later
        self.calctime = os.path.join(self.topdir, 'calctimer.txt')
        self.plottime = os.path.join(self.topdir, 'plottimer.txt')

        readme = open(os.path.join(self.topdir,'README.md'), 'w' )
        readme.write(repr(readme_dict))
        readme.close()

        print('done setting up metaparameters')
