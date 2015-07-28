import math as m
import sys
import numpy as np
import os

class persurv(object):

  def __init__(self,meta,p_run,s):

    #set true value of N(z) for this survey size
    self.seed = meta.survs[s]
    self.trueNz = self.seed*p_run.realistic_pdf
    self.logtrueNz = [m.log(max(self.trueNz[k],sys.float_info.epsilon)) for k in p_run.binnos]

    #define flat distribution for N(z)
    self.flat = self.seed*p_run.avgprob
    self.logflat = m.log(self.flat)
    self.flatNz = np.array([self.flat]*p_run.nbins)
    self.logflatNz = np.array([self.logflat]*p_run.nbins)

    self.topdir_s = p_run.topdir_p+'/'+str(self.seed)
    if not os.path.exists(self.topdir_s):
      os.makedirs(self.topdir_s)
