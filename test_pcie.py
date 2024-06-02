from datetime import datetime

import jax
import numpy as np

dt1 = datetime.now()
val = np.ones([32768,32768])

print (datetime.now() -dt1)

dt2 = datetime.now()
val = np.ones([65536,65536])

print(datetime.now() - dt2)


