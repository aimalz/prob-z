"""
perparam module permits varying of number of parameters defining N(z)
"""

import numpy as np
import sys
import math as m
import os
from util import path
import key

# define the class for parameter test
class perparam(object):

    def __init__(self, meta, p):
        self.p = p
        path_builder = path("{topdir}/{p}")
        self.path_builder = path_builder.fill(topdir=meta.topdir, p=self.p)
        self.key = key.key(p=p)

        # perparam differs from meta
        self.ndims = meta.params[self.p]
        self.allzs = meta.allzs[:self.ndims+1]#sorted(set(self.zlos+self.zhis))
        self.zlos = self.allzs[:-1]#meta.allzlos[:self.ndims]
        self.zhis = self.allzs[1:]#meta.allzhis[:self.ndims]
        self.zmids = (self.zlos+self.zhis)/2.
        self.zavg = sum(self.zmids)/self.ndims

        # define realistic underlying P(z) for this number of parameters
        self.realsum = sum(meta.realistic[:self.ndims])
        self.realistic_pdf = np.array([meta.realistic[k]/self.realsum/meta.zdifs[k] for k in xrange(0,self.ndims)])
        self.truePz = self.realistic_pdf
        self.logtruePz = np.array([m.log(max(tPz,sys.float_info.epsilon)) for tPz in self.truePz])

        # define flat P(z) for this number of parameters
        self.avgprob = 1./self.ndims/meta.zdif
        self.logavgprob = m.log(self.avgprob)
        self.flatPz = [self.avgprob]*self.ndims
        self.logflatPz = [self.logavgprob]*self.ndims

        print('initialized '+str(self.ndims)+' parameter test')

    # associated directory for test of this number of parameters
    def get_dir(self):
        return self.path_builder.construct()

