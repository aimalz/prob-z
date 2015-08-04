import os
import hickle as hkl

class perinit(object):

  def __init__(self,meta,p_run,s_run,n_run,i):#input initno

    self.p = p_run.p
    self.s = s_run.s
    self.n = n_run.n
    self.i = i
    self.init = meta.inits[i]
    #set up directory structure
    self.topdir_i = n_run.topdir_n+'/'+self.init
    if not os.path.exists(self.topdir_i):
      os.makedirs(self.topdir_i)
    self.fitness = os.path.join(self.topdir_i,'fit.txt')#[[[inpaths[s][n][t]+'fitness.p' for t in testnos] for n in sampnos] for s in survnos]

    self.topdirs_o = []
    for outdir in meta.outdirs:
      topdir_o = self.topdir_i+'/'+outdir
      self.topdirs_o.append(topdir_o)
      if not os.path.exists(topdir_o):
        os.makedirs(topdir_o)

    #generate initial values for walkers
    if self.init == 'ps':
      self.iguesses,self.mean = n_run.priordist.sample_ps(n_run.nwalkers)

    elif self.init == 'gm':
      self.iguesses,self.mean = n_run.priordist.sample_gm(n_run.nwalkers)

    elif self.init == 'gs':
      self.iguesses,self.mean = n_run.priordist.sample_gs(n_run.nwalkers)

    self.sampler = n_run.sampler()

    #self.filename = [topdirs_o+'/'+str(x)+'.h' for x in meta.plot_iters]
    #self.outnames = [[os.path.join(i_run.topdir_i,meta.outnames[t][r]) for r in meta.stepnos] for t in statnos]
