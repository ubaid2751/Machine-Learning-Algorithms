import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-100, 100, 100)

y = x / (1 + np.exp(-x))

plt.plot(x, y)
plt.show()