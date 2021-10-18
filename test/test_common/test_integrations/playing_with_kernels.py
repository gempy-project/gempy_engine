#%%
import numpy as np

d = 7
a = 3
print(np.exp(-(d**2/(2*a**2))))

print((d**2/(2*a**2)))
#%%
def exp_f(d, a):
    return np.exp(-(d**2/2*a**2))

exp_f(10, 5)
