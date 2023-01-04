import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))
x = np.linspace(-1, 1)
y = np.sin(np.pi/2 * x)
plt.xlabel('x')
plt.ylabel('sin(pi/2 x)')
plt.plot(x, y, '-k')
plt.show()