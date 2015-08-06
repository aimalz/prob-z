import math as m
import sys
import numpy as np
import os
#import matplotlib.pyplot as plt

class persurv(object):

  def __init__(self,meta,p_run,s):

    self.p = p_run.p
    self.s = s
    #set true value of N(z) for this survey size
    self.seed = meta.survs[self.s]
    self.trueNz = self.seed*p_run.realistic_pdf
    self.logtrueNz = [m.log(max(self.trueNz[k],sys.float_info.epsilon)) for k in p_run.dimnos]

    #define flat distribution for N(z)
    self.flat = self.seed*p_run.avgprob
    self.logflat = m.log(self.flat)
    self.flatNz = np.array([self.flat]*p_run.ndims)
    self.logflatNz = np.array([self.logflat]*p_run.ndims)

    self.topdir_s = p_run.topdir_p+'/'+str(self.seed)
    if not os.path.exists(self.topdir_s):
      os.makedirs(self.topdir_s)

#   def plot_setup(self):
#     f = plt.figure(figsize=(5,5))#*nsurvs,5))#plt.subplots(1, nsurvs, figsize=(5*nsurvs,5))
#     #print 'created subplots'
#     #for s in survnos:
#     sps = f.add_subplot(1,1,1)#,nsurvs,s+1)
#     sps.set_title(r'True $N(z)$ for '+str(s_run.seed)+' galaxies')
#     sps.set_xlabel(r'binned $z$')
#     sps.set_ylabel(r'$\ln N(z)$')
#     #sps.hlines(logtrueNz[s],zlos,zhis,color='k',linestyle='--',label=r'true $\ln N(z)$')
#     sps.step(meta.zmids,s_run.logflatNz,color='k',label=r'flat $\ln N(z)$',where='mid')
#     # thing.step(zmids,[-10]*ndims,color='k',label=r'$J='+str(seed_ngals[s])+r'$')

#   def plot_wrapup(self):
#     sps.set_ylim((-1.,ymax[s]+1.))
#     sps.legend(loc='upper left',fontsize='x-small')
#     f.savefig(os.path.join(s_run.topdir_s,'trueNz.png'))

    print('initialized '+str(self.seed)+' galaxy survey')
