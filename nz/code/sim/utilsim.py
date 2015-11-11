"""
util-sim module defines handy tools used in data generation
"""

import sys
import numpy as np
#import random
import bisect

np.random.seed(seed=0)

def lrange(l):
    """
    lrange(l) makes a range based on the length of a list or array l
    """
    return xrange(len(l))

# tools for sampling an arbitrary distribution, used in data generation
def cdf(weights):
    """
    cdf takes weights and makes them a normalized CDF
    """
    tot = sum(weights)
    result = []
    cumsum = 0.
    for w in weights:
      cumsum += w
      result.append(cumsum/tot)
    return result

def choice(pop, weights):
    """
    choice takes a population and assigns each element a value from 0 to len(weights) based on CDF of weights
    """
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = np.random.random()
    index = bisect.bisect(cdf_vals,x)
    return pop[index]
