import numpy as np

a=np.random.randn(3, 3) # a.shape=(3,3)a.shape=(3,3)

b=np.random.randn(3, 3) # b.shape=(2,1)b.shape=(2,1)

c=a**2 + b.T**2
print(c)