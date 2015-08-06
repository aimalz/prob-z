import numpy as np
import sys
import math as m
import os

class perparam(object):

  def __init__(self,meta,p):

    self.p = p
    self.ndims = meta.params[self.p]
    self.dimnos = range(0,self.ndims)
    self.allzs = meta.allzs[:self.ndims+1]#sorted(set(self.zlos+self.zhis))
    self.zlos = self.allzs[:-1]#meta.allzlos[:self.ndims]
    self.zhis = self.allzs[1:]#meta.allzhis[:self.ndims]
    #use centers of bins for plotting
    self.zmids = (self.zlos+self.zhis)/2.#[(self.zlos[k]+self.zhis[k])/2. for k in self.dimnos]
    self.zavg = sum(self.zmids)/self.ndims
    #self.zdifs = [self.zhis[k]-self.zlos[k] for k in self.dimnos]
    #self.zdif = sum(self.zdifs)/self.ndims

    #set true values of P(z) for this number of parameters
    self.realsum = sum(meta.realistic[:self.ndims])
    self.realistic_pdf = np.array([meta.realistic[k]/self.realsum/meta.zdifs[k] for k in self.dimnos])
    self.truePz = self.realistic_pdf#[realistic_pdf for s in survnos]
    self.logtruePz = np.array([m.log(max(self.truePz[k],sys.float_info.epsilon)) for k in self.dimnos])

    self.avgprob = 1./self.ndims/meta.zdif
    self.logavgprob = m.log(self.avgprob)
    self.flatPz = [self.avgprob]*self.ndims
    self.logflatPz = [self.logavgprob]*self.ndims

    self.topdir_p = meta.topdir+'/'+str(self.ndims)
    if not os.path.exists(self.topdir_p):
      os.makedirs(self.topdir_p)

#meta.queues[0].put()
