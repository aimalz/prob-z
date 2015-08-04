import os
import cPickle
import timeit

import pyfits

import math as m
import numpy as np

import multiprocessing as mp

#binurl = "http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits"
#binloc = os.path.join('../big-data/','zbins-12.fits')
#realcomps = [(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]


class meta(object):

  def __init__(self):#,loc=binloc,real=realcomps,survs=):

    if os.path.isfile('topdir.p'):
      self.topdir = cPickle.load(open('topdir.p','rb'))
    else:
      self.topdir = 'test'+str(round(timeit.default_timer()))
      cPickle.dump(self.topdir,open('topdir.p','wb'))
      os.makedirs(self.topdir)
    print('topdir='+str(self.topdir))

    self.loc = "http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits"
    print('loc='+str(self.loc))

    #set up for probability distributions
    #all p(z) share these bins from Sheldon, et al.

    #zbins = pyfits.open(os.path.join('../big-data/','zbins-12.fits'))
    zbins = pyfits.open(self.loc)#"http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits")

    #reformat bins
    #maxbins = len(zbins[1].data)
    self.allnbins = len(zbins[1].data)
    #nbins = maxbins
    self.allbinnos = range(0,self.allnbins)
    #binnos = [range(0,K) for K in nbins]
    self.allzlos = np.array([zbins[1].data[k][0] for k in self.allbinnos])
    self.allzhis = np.array([zbins[1].data[k][1] for k in self.allbinnos])
    #allzs = sorted(set(zlos+zhis))

    zbins.close()

    #use centers of bins for plotting
    self.allzmids = (self.allzhis+self.allzlos)/2.
    #zavg = sum(zmids)/nbins

    #useful for plotting
    self.zdifs = self.allzhis-self.allzlos
    self.zdif = sum(self.zdifs)/self.allnbins

    self.real = [(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]
    print('real='+str(self.real))
    self.nreal = len(self.real)
    self.realnos = range(0,self.nreal)

    #set true value of P(z)
    #tuples of form z_center, spread,magnitude
    #self.realistic_prep = real#[(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]
    self.realistic_comps = np.transpose([[zmid*tup[2]*(2*m.pi*tup[1])**-0.5*m.exp(-(zmid-tup[0])**2/(2*tup[1])) for zmid in self.allzmids] for tup in self.real])
    self.realistic = np.array([sum(self.realistic_comps[k]) for k in self.allbinnos])

    #set up for all tests
    #numbers of parameters
    self.params = [self.allnbins]
    print('params='+str(self.params))
    self.nparams = len(self.params)
    self.paramnos = range(0,self.nparams)

    #generate number of galaxies to draw
    #for consistency, must have more than one survey size
    #seed_ngals = [2,20]#2*np.arange(1,6)#[1,10]#can generate for different survey sizes
    self.survs = [2,20]
    print('survs='+str(self.survs))
    self.nsurvs = len(self.survs)
    self.survnos = range(0,self.nsurvs)

    #nsamps = 1#instantiations of the survey, more than 1 breaks some things...
    self.samps = 1
    print('samps='+str(self.samps))
    #self.nsamps = self.samps
    self.sampnos = range(0,self.samps)

    #initialization conditions for MCMC
    self.inits = ['ps/','gm/','gs/']
    print('inits='+str(self.inits))
    self.ninits = len(self.inits)
    self.initnos = range(0,self.ninits)

    self.colors = 'rgbymc'
    print('colors='+str(self.colors))
    self.ncolors = len(self.colors)
    self.colornos = range(0,self.ncolors)

#MVN prior as class
class mvn(object):

    def __init__(self,mean,cov):
        self.dims = len(mean)
        self.mean = mean
        self.cov = cov
        #assert np.all(sp.linalg.eigh(self.cov)[0] >= 0.)
        self.icov = np.linalg.pinv(self.cov,rcond = sys.float_info.epsilon)
        #assert np.all(sp.linalg.eigh(self.icov)[0] >= 0.)
        (self.logdetsign,self.logdet) = np.linalg.slogdet(self.cov)
        #self.logdet = np.log(-1.*np.linalg.det(self.cov))
        #assert self.logdetsign >= 0.

    def logpdf(self,x):
        delta = x-self.mean
        c = np.dot(delta, np.dot(self.icov, delta))
        prob = -0.5*c
        return prob

    def sample_ps(self,W):
        #outsamp = [sp.stats.multivariate_normal.rvs(mean=self.mean,cov=self.cov) for n in range(0,N)]
        outsamp = [thing for thing in np.array(np.random.multivariate_normal(self.mean,self.cov,W))]
        #sampprobs = [self.logpdf(s) for s in outsamp]
        #assert np.any(np.isnan(sampprobs)) == False
        return outsamp#,sampprobs

    def sample_gm(self,W):
        #rando = sp.stats.multivariate_normal.rvs(mean=self.mean,cov=self.cov)
        #sigma = 100.#[np.median(self.cov[i]) for i in range(0,dims)]
        outsamp = [self.mean+np.random.randn(self.dims) for w in range(0,W)]#[rando+np.random.randn(dims) for n in range(0,N)]
        #sampprobs = [self.logpdf(s) for s in outsamp]
        #assert np.any(np.isnan(sampprobs)) == False
        return outsamp#,sampprobs

    def sample_gs(self,W):
        #rando = sp.stats.multivariate_normal.rvs(mean=self.mean,cov=self.cov)
        rando = np.random.multivariate_normal(self.mean,self.cov)
        #rando = self.sample_dumb(1)[0]
        #sigma = 100.#[np.median(self.cov[i]) for i in range(0,dims)]
        outsamp = [rando+np.random.randn(self.dims) for w in range(0,W)]#[rando+np.random.randn(dims) for n in range(0,N)]
        #sampprobs = [self.logpdf(s) for s in outsamp]
        #assert np.any(np.isnan(sampprobs)) == False
        return outsamp,rando#,sampprobs,rando

#   def plot(self):

#     realsum = sum(self.realistic)
#     realistic_pdf = self.realistic/self.zdifs/realsum
#     plotrealistic = np.array([sum(r) for r in self.realistic_comps])
#     plotrealisticsum = sum(self.realistic_comps)
#     plotrealistic_comps = np.transpose(self.realistic_comps/plotrealisticsum)
#     plotrealistic_pdf = plotrealistic/plotrealisticsum

#     #print 'plot_priorgen'
#     f = plt.figure(figsize=(5,5))
#     #print 'one'
#     sys.stdout.flush()
#     sps = f.add_subplot(1,1,1)
#     f.suptitle('True p(z)')# for $J=$'+str(ngals_seed))
#     sps.step(self.allzmids,plotrealistic_pdf,c='k',label='True p(z)')
#     for r in self.realnos:
#       sps.step(self.allzmids,plotrealistic_comps[r],c=self.colors[r],label='component '+str(real[r][2])+'N('+str(real[r][0])+','+str(real[r][1])+')')
#     sps.set_ylabel('p(z)')
#     sps.set_xlabel('z')
#     sps.legend(fontsize='x-small',loc='upper left')
#     f.savefig(os.path.join(self.topdir,'physPz.png'))
#     #print 'done?'
#     return
