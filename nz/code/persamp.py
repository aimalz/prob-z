import os
import random
import bisect
import cPickle
import numpy as np
import sys
import math as m
import scipy as sp
from scipy import stats

def cdf(weights):
    tot = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
      cumsum += w
      result.append(cumsum/tot)
    return result

def choice(pop, weights):
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    index = bisect.bisect(cdf_vals,x)
    return pop[index]

class persamp(object):

  def __init__(self,meta,p_run,s_run,n):

    #sample some number of galaxies
    self.nsamps = meta.samps
    self.ngals = s_run.seed#nsamps*[seed]
   #for poisson sampling instead of set survey size -- small number tests fail when sample size is 0!
   #ngals = np.random.poisson(seed)#[[np.random.poisson(seed) for n in sampnos] for s in survnos]
    self.galnos = range(0,self.ngals)
    #print([j for j in self.galnos])

    self.topdir_n = s_run.topdir_s+'/'+str(n)+'-'+str(self.ngals)
    if not os.path.exists(self.topdir_n):
      os.makedirs(self.topdir_n)

    def maketrue(self):

      if os.path.exists(self.topdir_n+'true.p'):
        truefile = open(self.topdir_n+'true.p','rb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        [count,(sampPz,logsampPz),(sampNz,logsampNz)] = cPickle.load(truefile)
        truefile.close()
      else:
      ##randomly select bin counts from true N(z)
      # count = [0]*nbins
      # for i in range(0,ngals):
      #     count[choice(binnos, truePz)] += 1
      #
      #plotcounts = [[[max(bincounts[s][n][k]/zdifs[k],sys.float_info.epsilon) for k in binnos] for n in sampnos] for s in survnos]
      #logplotcounts = np.log(np.array(plotcounts))

      #test case: all galaxies in survey have same true redshift
        chosenbin = np.argmax(p_run.truePz)#choice(binnos, realistic_pdf)#random.sample(range(0,35),1)#np.argmax(plotrealistic_pdf)
        count = [0]*p_run.nbins
        count[chosenbin] = self.ngals
        count = np.array(count)

      #plotcounts = [[[max(bincounts[s][n][k]/zdifs[k],sys.float_info.epsilon) for k in binnos] for n in sampnos] for s in survnos]
      #logplotcounts = np.log(np.array(plotcounts))

        sampNz = count/p_run.zdif
        logsampNz = np.log(np.array([max(o,sys.float_info.epsilon) for o in sampNz]))

      #sampNz = np.array([[bincounts[s][n]/zdif for n in sampnos] for s in survnos])
      #logsampNz = np.log(np.array([[[max(o,sys.float_info.epsilon) for o in counts] for counts in samp] for samp in sampNz]))

        sampPz = sampNz/self.ngals#count/ngal/zdif
        logsampPz = np.log(np.array([max(o,sys.float_info.epsilon) for o in sampPz]))

      #sampPz = np.array([[bincounts[s][n]/ngals[s][n]/zdif for n in sampnos] for s in survnos])
      #logsampPz = np.log(np.array([[[max(o,sys.float_info.epsilon) for o in counts] for counts in samp] for samp in sampPz]))

      #for s in survnos:
      #  for n in sampnos:
        truefile = open(self.topdir_n+'true.p','wb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        cPickle.dump([count,(sampPz,logsampPz),(sampNz,logsampNz)],truefile)
        truefile.close()

      return([count,(sampPz,logsampPz),(sampNz,logsampNz)])

    [self.count,(self.sampPz,self.logsampPz),(self.sampNz,self.logsampNz)] = maketrue(self)

    #generate the catalog of individual galaxy posteriors
    def makecat(self):

      if os.path.exists(self.topdir_n+'datacat.p'):
        catfile = open(self.topdir_n+'datacat.p','rb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        [shiftZs,sigZs] = cPickle.load(catfile)
        catfile.close()
      else:
      #  #turn bin numbers into redshifts for histogram later
      #  idealZs = np.array([[zmids[k] for k in binnos] for j in range(0,int(round(trueNz[k])))])

      #assign actual redshifts uniformly within each bin
      #trueZs = ([random.uniform(zlos[k],zhis[k]) for k in binnos for j in range(0,count[k])])

      #test case: all galaxies have same true redshift
        trueZs = np.array([p_run.zmids[k] for k in p_run.binnos for j in range(0,self.count[k])])

      #jitter zs to simulate inaccuracy
        modZs = trueZs+1.#[[trueZs[s][n]+1. for n in sampnos] for s in survnos]
        varZs = [p_run.zdif*modZs[j] for j in self.galnos]# for n in sampnos] for s in survnos])#zdif*(trueZs+1.)
        shiftZs = np.array([random.gauss(trueZs[j],varZs[j]) for j in self.galnos])
        sigZs = np.array([abs(random.gauss(varZs[j],varZs[j])) for j in self.galnos])
      #print(modZs,varZs,sigZs)
      #pdb.set_trace()

      #test case: perfect observations
      #shiftZs = trueZs
      #sigZs = np.array([[[zdif*(trueZs[s][n][j]+1.) for j in range(0,ngals[s][n])] for n in sampnos] for s in survnos])

      #write out the data into a "catalog"
      #broken re: nsamps>1
        catfile = open(self.topdir_n+'datacat.p','wb')#open(str(n+1)+'datacat'+str(seed_ngals[s])+'.p','w')
        catdat = [shiftZs,sigZs]
        cPickle.dump(catdat,catfile)
        catfile.close()

      return(shiftZs,sigZs)

    self.obsZs,self.obserrs = makecat(self)

    def setup_pdfs(self):
    #redefine bins
      minobs = min(self.obsZs)
      maxobs = max(self.obsZs)
      binfront = [min(p_run.zlos)+x*p_run.zdif for x in range(int(m.floor((minobs-min(p_run.zlos))/p_run.zdif)),0)]
      binback = [max(p_run.zhis)+x*p_run.zdif for x in range(1,int(m.ceil((maxobs-max(p_run.zhis))/p_run.zdif)))]
      binends = np.array(binfront+sorted(set(p_run.zlos+p_run.zhis))+binback)
      binlos = binends[:-1]
      binhis = binends[1:]
      new_nbins = len(binends)-1
      new_binnos = range(0,new_nbins)
      binmids = (binhis+binlos)/2.#[(binends[k]+binends[k+1])/2. for k in new_binnos]

      return([binends,new_nbins,new_binnos])

    self.binends,self.new_nbins,self.new_binnos = setup_pdfs(self)

    def makepdfs(self):

    #generate gaussian likelihood function per galaxy per sample per survey to simulate imprecision
    #simultaneously generate sheldon "posterior"
      pobs = []
      logpobs = []
      for j in self.galnos:
        func = sp.stats.norm(loc=self.obsZs[j],scale=self.obserrs[j])
        lo = np.array([max(sys.float_info.epsilon,func.cdf(self.binends[k])) for k in self.new_binnos])
        hi = np.array([max(sys.float_info.epsilon,func.cdf(self.binends[k+1])) for k in self.new_binnos])
        spread = (hi-lo)/p_run.zdif
      #normalize probabilities to integrate (not sum)) to 1
        summed = sum(spread)
        pob = spread/summed/p_run.zdif
        logpob = [m.log(max(p_i,sys.float_info.epsilon)) for p_i in pob]
        logpobs.append(logpob)
        pobs.append(pob)
      pobs = np.array(pobs)
      logpobs = np.array(logpobs)
      sheldonprep = np.sum(np.array(pobs),axis=0)
      sheldon = [max(sys.float_info.epsilon,sheldonprep[k]) for k in self.new_binnos]
      logsheldon = np.log(np.array(sheldon))

      return([pobs,logpobs],[sheldon,logsheldon])

    [self.pobs,self.logpobs],[self.sheldon,self.logsheldon] = makepdfs(self)
