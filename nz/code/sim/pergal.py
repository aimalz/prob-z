"""
per-gal module defines class of one galaxy
"""
from collections import namedtuple
class pergal(object):
    """
    pergal class is one galaxy
    """

    def __init__(self, meta, ind = None):

        self.meta = meta

        self.ind = ind

        self.ncomps = 1
        self.nouts = 0

        self.compvar = []
        self.outvar = []
        self.outobs = []

        self.truZ = None
        self.zfactor = 1.

        self.sigZ = []
        self.shift = []

        self.obsZ = []

        self.elements = []
        self.outelems = []

        self.distdgen = None
        self.const = 0.

        #parametrized posterior distribution
        self.mapZ = None
        self.expZ = None
        self.post = []
        self.logpost = []

    def makezfactor(self,tru):
        if self.meta.sigma == True:
            zfactor = (1.+tru)**self.meta.noisefact
        else:
            zfactor = 1.
        return zfactor

ObsStats = namedtuple("ObsStats", ['shift', 'stddev', 'weight'])
OutStats = namedtuple("OutStats", ['obsZ', 'stddev', 'weight'])
