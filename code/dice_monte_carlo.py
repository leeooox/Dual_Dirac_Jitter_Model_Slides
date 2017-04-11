import numpy as np
from numpy.random import random_integers
import matplotlib.pyplot as plt

sample_len = 1000000
res = np.zeros(sample_len)
for i in range(sample_len):
    res[i] = np.sum(random_integers(1,6,7))
plt.hist(res,bins=35,normed=True)
plt.show()
