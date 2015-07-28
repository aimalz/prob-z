import numpy as np
import sys
import math as m
import os

class perparam(object):

  def __init__(self,meta,p):

    self.nbins = meta.params[p]
    self.binnos = range(0,self.nbins)
    self.zlos = meta.allzlos[:self.nbins]
    self.zhis = meta.allzhis[:self.nbins]
    self.allzs = sorted(set(self.zlos+self.zhis))
    #use centers of bins for plotting
    self.zmids = [(self.zlos[k]+self.zhis[k])/2. for k in self.binnos]
    self.zavg = sum(self.zmids)/self.nbins
    self.zdifs = [self.zhis[k]-self.zlos[k] for k in self.binnos]
    self.zdif = sum(self.zdifs)/self.nbins

    #set true values of P(z) for this number of parameters
    self.realsum = sum(meta.realistic[:self.nbins])
    self.realistic_pdf = np.array([meta.realistic[k]/self.realsum/self.zdifs[k] for k in self.binnos])
    self.truePz = self.realistic_pdf#[realistic_pdf for s in survnos]
    self.logtruePz = np.array([m.log(max(self.truePz[k],sys.float_info.epsilon)) for k in self.binnos])

    self.avgprob = 1./self.nbins/self.zdif
    self.logavgprob = m.log(self.avgprob)
    self.flatPz = np.array([self.avgprob]*self.nbins)
    self.logflatPz = np.log(self.flatPz)

    self.topdir_p = meta.topdir+'/'+str(self.nbins)
    if not os.path.exists(self.topdir_p):
      os.makedirs(self.topdir_p)
