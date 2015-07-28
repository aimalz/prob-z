import meta
from meta import *

meta = meta()

import perparam
from perparam import *

import persurv
from persurv import *

import persamp
from persamp import *

for p in meta.paramnos:
  p_run = perparam(meta,p)

  for s in meta.survnos:
    s_run = persurv(meta,p_run,s)

    for n in meta.sampnos:
      n_run = persamp(meta,p_run,s_run,n)

#      for i in meta.initnos:
#        i_run = perinit(meta,p_run,s_run,n_run)
