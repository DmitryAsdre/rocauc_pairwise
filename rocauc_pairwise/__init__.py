import os
import numpy as np

os.environ["C_INCLUDE_PATH"] = np.get_include()

import pyximport
pyximport.install()