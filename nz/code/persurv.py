# one of these objects per approximate survey size

import math as m
import sys
import numpy as np
import os
from util import path
import key

# define class for survey size test
class persurv(object):

    path_builder = path("{topdir}/{p}/{s}")

    def __init__(self,meta,p_run,s):
        self.p_run = p_run
        self.p = p_run.p
        self.s = s
        self.path_builder = persurv.path_builder.fill(topdir=meta.topdir, p=p_run.p, s=s)
        self.key = p_run.key.add(s=s)

        # set true value of N(z) for this survey size
        self.seed = meta.survs[self.s]
        self.trueNz = self.seed*p_run.realistic_pdf
        self.logtrueNz = [m.log(max(x,sys.float_info.epsilon)) for x in self.trueNz]

        # define flat distribution for N(z)
        self.flat = self.seed*p_run.avgprob
        self.logflat = m.log(self.flat)
        self.flatNz = np.array([self.flat]*p_run.ndims)
        self.logflatNz = np.array([self.logflat]*p_run.ndims)
        self.n_runs = []

        #     # parameters for mcmc, used to need these, keeping around in case I missed one
        #     self.miniters = self.seed**2#int(1e3)
        #     self.nsteps = self.p#self.maxiters/self.miniters
        #     self.maxiters = self.miniters*self.p#int(1e4)
        #     self.stepnos = np.arange(0,self.nsteps)
        #     self.iternos = self.miniters*(self.stepnos+1)
        #     self.oneiternos = range(0,self.miniters)
        #     self.alliternos = range(0,self.maxiters)
        #     self.eachiternos = [self.miniters*np.arange(self.stepnos[r]-1,self.stepnos[r]) for r in self.stepnos]
        #     self.thinto = self.seed#int(1e2)
        #     self.ntimes = self.miniters/self.thinto
        #     self.timenos = range(0,self.ntimes)
        #     self.alltimenos = range(0,self.maxiters/self.thinto,self.thinto)
        #     self.eachtimenos = [range(self.iternos[r]-self.miniters,self.iternos[r],self.thinto) for r in self.stepnos]
        #     self.filenames = [str(ins)+'.h' for ins in self.iternos]
        #     self.outnames = [[t+'/'+r for r in self.filenames] for t in self.outdirs]

        print('initialized '+str(self.seed)+' galaxy survey')

    # associated directory for this survey size
    def get_dir(self):
        return self.path_builder.construct()
