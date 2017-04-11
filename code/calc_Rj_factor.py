from scipy.special import erfcinv
import numpy as np

BER = 1E-12
print 2*np.sqrt(2)*erfcinv(2*BER)
