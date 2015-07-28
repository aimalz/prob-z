import pyfits

import math as m
import numpy as np

import os
import cPickle
import timeit

class meta(object):

  loc = "http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits"
  print('loc='+str(loc))

  #set up for probability distributions
  #all p(z) share these bins from Sheldon, et al.

  #self.loc = loc
  #zbins = pyfits.open(os.path.join('../big-data/','zbins-12.fits'))
  zbins = pyfits.open(loc)#"http://data.sdss3.org/sas/dr8/groups/boss/photoObj/photoz-weight/zbins-12.fits")

  #reformat bins
  #maxbins = len(zbins[1].data)
  allnbins = len(zbins[1].data)
  #nbins = maxbins
  allbinnos = range(0,allnbins)
  #binnos = [range(0,K) for K in nbins]
  allzlos = [zbins[1].data[k][0] for k in allbinnos]
  allzhis = [zbins[1].data[k][1] for k in allbinnos]
  #allzs = sorted(set(zlos+zhis))

  zbins.close()

  #use centers of bins for plotting
  allzmids = [(allzhis[k]+allzlos[k])/2. for k in allbinnos]
  #zavg = sum(zmids)/nbins

  #useful for plotting
  #zdifs = [zbins[1].data[i][1]-zbins[1].data[i][0] for i in binnos]
  #zdif = sum(zdifs)/nbins

  real = [(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]
  print('real='+str(real))

  #set true value of P(z)
  #tuples of form z_center, spread,magnitude
  realistic_prep = real#[(0.2,0.005,2.0),(0.4,0.005,1.25),(0.5,0.1,2.0),(0.6,0.005,1.25),(0.8,0.005,1.25),(1.0,0.005,0.75)]
  realistic_comps = np.transpose([[zmid*tup[2]*(2*m.pi*tup[1])**-0.5*m.exp(-(zmid-tup[0])**2/(2*tup[1])) for zmid in allzmids] for tup in realistic_prep])
  realistic = np.array([sum(realistic_comps[k]) for k in allbinnos])

  if os.path.isfile('topdir.p'):
    topdir = cPickle.load(open('topdir.p','rb'))
    print('topdir='+str(topdir))
  else:
    topdir = 'test'+str(round(timeit.default_timer()))
    print('topdir='+str(topdir))
    cPickle.dump(topdir,open('topdir.p','wb'))
    os.makedirs(topdir)

  #set up for all tests
  #numbers of parameters
  params = [allnbins]
  print('params='+str(params))

  nparams = len(params)
  paramnos = range(0,nparams)

  #generate number of galaxies to draw
  #for consistency, must have more than one survey size
  #seed_ngals = [2,20]#2*np.arange(1,6)#[1,10]#can generate for different survey sizes
  survs = [2,20]
  print('survs='+str(survs))

  nsurvs = len(survs)
  survnos = range(0,nsurvs)

  #nsamps = 1#instantiations of the survey, more than 1 breaks some things...
  samps = 1
  print('samps='+str(samps))

  nsamps = samps
  sampnos = range(0,nsamps)

  #initialization conditions for MCMC
  inits = ['ps/','gm/','gs/']
  print('inits='+str(inits))

  ninits = len(inits)
  initnos = range(0,ninits)

  colors = 'rgbymc'
  print('colors='+str(colors))

  ncolors = len(colors)
  colornos = range(0,ncolors)
