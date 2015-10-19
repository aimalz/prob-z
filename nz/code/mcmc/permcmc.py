"""
per-mcmc module is variable object for MCMC sub-runs
"""

import os
import statistics
from subprocess import call
import timeit
import psutil
import statmcmc as stats
from utilmcmc import *
import keymcmc as key

# test whether burning in or done with that, true if burning in
def burntest(outputs,run):#output,r_run):# of dimensions nwalkers*miniters
    print('testing burn-in condition')
    lnprobs = outputs['probs']
    lnprobs = np.swapaxes(lnprobs,0,1).T
    varprob = sum([statistics.variance(w) for w in lnprobs])/run.meta.nwalkers
    difprob = statistics.median([(lnprobs[w][0]-lnprobs[w][-1])**2 for w in xrange(run.meta.nwalkers)])
    if difprob > varprob:
        print('burning-in '+str(difprob)+' > '+str(varprob))
    else:
        print('post-burn '+str(difprob)+' < '+str(varprob))
    return(difprob > varprob)

# define class per initialization of MCMC
# will be changing this object at each iteration as it encompasses state
class pertest(object):

    def __init__(self, meta):

        self.meta = meta
        self.vals = self.meta.ivals
        self.sampler = self.meta.sampler
        self.burns = 0
        self.runs = 0

        with open(self.meta.calctime,'w') as calctimer:
            calctimer.write(str(timeit.default_timer())+' icalc \n')
            calctimer.close()

        # what outputs of emcee will we be saving?
        self.stats = [ stats.stat_chains(self.meta),
                       stats.stat_probs(self.meta),
                       stats.stat_fracs(self.meta),
                       stats.stat_times(self.meta) ]

    # sample with emcee and provide output
    # TO DO: change emcee parameters to save less data to reduce i/o and storage
    def sampling(self):
        sampler = self.meta.sampler
        ivals = self.vals
        miniters = self.meta.miniters
        thinto = self.meta.thinto
        ntimes = self.meta.ntimes
        start_time = timeit.default_timer()
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(ivals,miniters,thin=thinto)
        ovals = [walk[-1] for walk in sampler.chain]
        chains = sampler.chain#nsteps*nwalkers*nbins
        probs = sampler.lnprobability#nsteps*nwalkers
        fracs = sampler.acceptance_fraction#nwalkers
        times = stats.acors(chains)#nwalkers#sampler.get_autocorr_time(window = ntimes/2)#nbins
        outputs = { 'times':times,
                    'fracs':fracs,
                    'probs':probs,
                    'chains':chains }
        elapsed = timeit.default_timer() - start_time
        self.vals = ovals
        self.outputs = outputs
        self.elapsed = elapsed

        return(outputs,elapsed)

    def sampnsave(self,r,burnin):

        self.key = self.meta.key.add(r=r, burnin=burnin)

        (outputs,elapsed) = self.sampling()

        self.key.store_state(self.meta.topdir, outputs)
        for stat in self.meta.stats:
            stat.update(self.outputs[stat.name])

        # store the iteration number, so everyone knows how many iterations have been completed
        self.key.store_iterno(self.meta.topdir, self.runs)
        #print ('s_run={}, dist={}'.format(self.s_run, self.s_run.dist))
        self.meta.dist.complete_chunk(self.key)

        # record time of calculation for monitoring progress
        with open(self.meta.calctime,'a') as calctimer:
            process = psutil.Process(os.getpid())
            calctimer.write(str(timeit.default_timer())+' '+str(self.key)+' '+str(elapsed)+' mem:'+str(process.get_memory_info())+'\n')
            calctimer.close()

        return outputs

    def preburn(self, b):
        outputs = self.sampnsave(b,burnin=True)
        return outputs

    # once burn-in complete, know total number of runs remaining
    def atburn(self, b, outputs):
        print(str(b*self.meta.miniters)+' iterations of burn-in for '+str(self.meta.topdir))
        self.nsteps = 2*(b+1)
        self.maxiters = self.nsteps*self.meta.miniters
        self.alliternos = range(0,self.maxiters)
        self.alltimenos = range(0,self.maxiters/self.meta.thinto,self.meta.thinto)
        self.key.store_state(self.meta.topdir, outputs)
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
        outputs = self.preburn(self.burns)
        print('zeroth run done')

        while burntest(outputs,self):
            yield self
            self.burns += 1
            self.runs += 1
            outputs = self.preburn(self.runs)
        else:
            self.atburn(self.runs, outputs)
            while self.runs <= self.nsteps:#2*self.burns+1:
                yield self
                self.runs += 1
                self.postburn(self.runs)

        return
