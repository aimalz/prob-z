"""
key module defines loading/saving/naming conventions
"""

# TO DO: redo state handling with dicts or generators rather than lists

from util import path
import distribute
import cPickle
import hickle as hkl
import os
import inputs as meta

# read/write cPickle
def safe_load_c(path, num_objs = None):
    print 'loading: ' + path
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        if num_objs is None:
            return cPickle.load(f)
        return [cPickle.load(f) for _ in xrange(num_objs)]
    f.close()
def safe_store_c(path, o):
    print 'storing: ' + path
    directory = path[:path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "wb") as f:
        cPickle.dump(o, f)
    f.close()

# read/write hickle
def safe_load_h(path, num_objs = None):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        if num_objs is None:
            return hkl.load(f)
        return [hkl.load(f) for _ in xrange(num_objs)]
    f.close()
def safe_store_h(path, o):
    print 'storing hkl:' + path
    directory = path[:path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        hkl.dump(o, f)
    f.close()

# define templates for path
true_builder = path("{topdir}/{p}/{s}/{n}/true.p")
cat_builder = path("{topdir}/{p}/{s}/{n}/cat.p")
ivals_builder = path("{topdir}/{p}/{s}/{n}/{i}/ivals.p")
state_builder = path("{topdir}/{p}/{s}/{n}/{i}/state-{r}.p")
iterno_builder = path("{topdir}/{p}/{s}/{n}/{i}/iterno.p")
statistics_builder = path("{topdir}/{p}/{s}/{n}/{i}/stat_{stat_name}.p")
data_builder = path("{topdir}/{p}/{s}/{n}/{i}/stat_{data_name}.hkl")

# all the dict handling here
class key(distribute.distribute_key):

    def __init__(self, **kv):
        self.p = None
        self.s = None
        self.n = None
        self.i = None
        self.r = None
        self.burnin = None
        self.update_dict(kv)

    def copy(self):
        ret = key()
        ret.p = self.p
        ret.s = self.s
        ret.n = self.n
        ret.i = self.i
        ret.r = self.r
        ret.burnin = self.burnin
        return ret

    def __hash__(self):
        return hash((self.p, self.s, self.n, self.i, self.r, self.burnin))

    def __eq__(self, other):
        if not isinstance(other, key):
            return False
        return (self.p == other.p and
                self.s == other.s and
                self.n == other.n and
                self.i == other.i and
                self.r == other.r and
                self.burnin == other.burnin)

    def __repr__(self):
        return ("KEY: " + str(self.to_dict()))

    def update_dict(self, d):
        if 'p' in d:
            self.p = d['p']
        if 's' in d:
            self.s = d['s']
        if 'n' in d:
            self.n = d['n']
        if 'i' in d:
            self.i = d['i']
        if 'r' in d:
            self.r = d['r']
        if 'burnin' in d:
            self.burnin = d['burnin']

    def add_dict(self, d):
        ret = self.copy()
        ret.update_dict(d)
        return ret

    def add(self, **d):
        ret = self.copy()
        ret.update_dict(d)
        return ret

    def filter(self, s):
        newKey = key()
        if 'p' in s:
            newKey.p = self.p
        if 's' in s:
            newKey.s = self.s
        if 'n' in s:
            newKey.n = self.n
        if 'i' in s:
            newKey.i = self.i
        if 'r' in s:
            newKey.r = self.r
        if 'burnin' in s:
            newKey.burnin = self.r
        return newKey

    def to_dict(self, d = None):
        retval = d
        if retval is None:
            retval = {}
        if self.p is not None:
            retval['p'] = self.p
        if self.s is not None:
            retval['s'] = self.s
        if self.n is not None:
            retval['n'] = self.n
        if self.i is not None:
            retval['i'] = self.i
        if self.r is not None:
            retval['r'] = self.r
        if self.burnin is not None:
            retval['burnin'] = self.burnin
        return retval

    # should break this up around here to use inheritance rather than lumping everything into one class

    # true redshifts of survey
    def load_true(self, topdir):
        filepath = true_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)
    def store_true(self, topdir, o):
        filepath = true_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath, o)

    # catalog of posteriors as data
    def load_cat(self, topdir):
        filepath = cat_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)
    def store_cat(self, topdir, o):
        filepath = cat_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath, o)

    # initial values for plotting
    def load_ivals(self, topdir):
        filepath = ivals_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)
    def store_ivals(self, topdir, o):
        filepath = ivals_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath, o)

    # state is mutable permcmc object at each stage
    def load_state(self, topdir):
        filepath = state_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)
    def store_state(self, topdir, o):
        filepath = state_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath, o)

    # need to know iterno for progress
    def load_iterno(self, topdir):
        filepath = iterno_builder.construct(**self.to_dict({'topdir':topdir}))
        return safe_load_c(filepath)
    def store_iterno(self, topdir, o):
        filepath = iterno_builder.construct(**self.to_dict({'topdir':topdir}))
        safe_store_c(filepath, o)

    # intermediate stats now stored as a series of pickled entries, rather than one large list
    # need to know length of list to read it
    def load_stats(self, topdir, name, size):
        filepath = statistics_builder.construct(**self.to_dict({'topdir':topdir, 'stat_name':name}))
        return safe_load_c(filepath, size)
    def add_stats(self, topdir, name, o):
        filepath = statistics_builder.construct(**self.to_dict({'topdir':topdir, 'stat_name':name}))
        with open(filepath, "ab") as f:
            cPickle.dump(o, f)

    def load_data(self, topdir):
        filepath = data_builder.construct(**self.to_dict({'topdir':topdir, 'data_name':name}))
        return safe_load_h(filepath, size)
    def store_data(self, topdir, o):
        filepath = data_builder.construct(**self.to_dict({'topdir':topdir, 'data_name':name}))
        safe_store_c(filepath, o)


