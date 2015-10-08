"""
permcmc module governs updating of state information at each step of MCMC
"""

# TO DO: investigate emcee settings for autocorrelation time

import os
import timeit
import numpy as np
import cPickle
#import emcee
#import StringIO
from subprocess import call
import hickle as hkl
import statistics
import psutil

# test whether burning in or done with that, true if burning in
def burntest(output,r_run):# of dimensions nwalkers*miniters
    print('testing burn-in condition')
    lnprobs = output['probs']
    lnprobs = np.swapaxes(lnprobs,0,1).T
    varprob = sum([statistics.variance(w) for w in lnprobs])/r_run.n_run.nwalkers
    difprob = statistics.median([(lnprobs[w][0]-lnprobs[w][-1])**2 for w in r_run.n_run.walknos])
    if difprob > varprob:
        print('burning-in '+str(difprob)+' > '+str(varprob))
    else:
        print('post-burn '+str(difprob)+' < '+str(varprob))
    return(difprob > varprob)

# # old version of saving state
# def savestate(r_run):
#   statefile = open(r_run.statename,'wb')
#   tmp_queues = r_run.meta.queues
#   tmp_sampler = r_run.sampler
#   r_run.sampler = 'NOPE'
#   r_run.meta.queues=[]

#   print ('I AM SAVING: ' + repr(r_run))
#   print ('DONE')
#   cPickle.dump(r_run,statefile)
#   r_run.meta.queues = tmp_queues
#   r_run.sampler = tmp_sampler
#   statefile.close()
#   print('saving state')
#   call(['md5', r_run.statename])
#   call(['wc', r_run.statename])
#   return

# object that is updated as part of perinit
class permcmc(object):

    def __init__(self,i_run):

        self.i_run = i_run
        self.n_run = i_run.n_run
        self.s_run = i_run.s_run
        self.p_run = i_run.p_run
        self.meta = i_run.meta
        self.key = i_run.key

        self.ivals = self.i_run.iguesses
        self.sampler = self.n_run.sampler
        self.burns = 0
        self.runs = 0

        with open(self.meta.calctime,'w') as calctimer:
            calctimer.write(str(timeit.default_timer())+' icalc \n')
            calctimer.close()

    # sample with emcee and provide output
    # TO DO: change emcee parameters to save less data to reduce i/o and storage
    def sampling(self):
        sampler = self.n_run.sampler
        ivals = self.ivals
        miniters = self.meta.miniters
        thinto = self.meta.thinto
        start_time = timeit.default_timer()
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(ivals,miniters,thin=thinto)
        ovals = [walk[-1] for walk in sampler.chain]
        times = sampler.get_autocorr_time(window = 10)#ndims
        fracs = sampler.acceptance_fraction#nwalkers
        probs = sampler.lnprobability#niters*nwalkers
        chains = sampler.chain#niters*nwalkers*ndims
        outputs = { 'times':times,
                    'fracs':fracs,
                    'probs':probs,
                    'chains':chains }
        elapsed = timeit.default_timer() - start_time
        self.ivals = ovals
        self.outputs = outputs
        self.elapsed = elapsed

        return (outputs,elapsed)

    def sampnsave(self,r,burnin):

        self.key = self.i_run.key.add(r=r, burnin=burnin)

        (outputs,elapsed) = self.sampling()

        self.key.store_state(self.meta.topdir, outputs)
        for stat in self.i_run.stats:
            stat.update(self.outputs[stat.name])

        # store the iteration number, so everyone knows how many iterations have been completed
        self.key.store_iterno(self.meta.topdir, self.runs)
        print ('s_run={}, dist={}'.format(self.s_run, self.s_run.dist))

        self.s_run.dist.complete_chunk(self.key)

        # record time of calculation for monitoring progress
        with open(self.meta.calctime,'a') as calctimer:
            process = psutil.Process(os.getpid())
            calctimer.write(str(timeit.default_timer())+' '+str(self.key)+' '+str(elapsed)+' mem:'+str(process.get_memory_info())+'\n')
            calctimer.close()

        return outputs

    def preburn(self, b):
        output = self.sampnsave(b,burnin=True)
        return output

    # once burn-in complete, know total number of runs remaining
    def atburn(self, b, output):
        print(str(b*self.meta.miniters)+' iterations of burn-in for '+str((self.p_run.p,self.s_run.s,self.n_run.n,self.i_run.i)))
        self.nsteps = 3*(b+1)
        self.maxiters = self.nsteps*self.meta.miniters
        self.alliternos = range(0,self.maxiters)
        self.alltimenos = range(0,self.maxiters/self.meta.thinto,self.meta.thinto)
        self.key.store_state(self.meta.topdir, output)
        self.key.store_iterno(self.meta.topdir, self.runs)
        return

    def postburn(self, p):
        self.sampnsave(p,burnin=False)
        return

    # sample until burn-in complete, then do twice that number of runs before finishing
    def samplings(self):
        # may add ability to pick up where left off
        # if os.path.exists(self.statename):
        #   self = self.state()
        # else:
        output = self.preburn(self.burns)
        print('zeroth run done')

        while burntest(output, self):
            yield self
            self.burns += 1
            self.runs += 1
            output = self.preburn(self.runs)
        else:
            self.atburn(self.runs, output)
            while self.runs <= self.nsteps:#2*self.burns+1:
                yield self
                self.runs += 1
                self.postburn(self.runs)

        return
