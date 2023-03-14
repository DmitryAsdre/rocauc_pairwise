import os
import numpy as np
#from os import environ
os.environ['CFLAGS'] = '-fopenmp'
os.environ['LDFLAGS'] = '-fopenmp'

os.environ["C_INCLUDE_PATH"] = np.get_include()

import pyximport
pyximport.install()