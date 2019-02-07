f = lambda x, y : x + y
print(f(1,1))


from scipy import stats
import numpy as np 
www = np.random.seed(12345678)
x = stats.norm.rvs(loc=5, scale=3, size=100)
print('\n\n')
print(x)
y = stats.shapiro(x)
# (0.9772805571556091, 0.08144091814756393)
print('\n\n')
print(y)