import matplotlib.pylab as plt
import numpy as np

with open('test.npy', 'rb') as f:
    traces = np.load(f)

plt.figure()
plt.plot(traces[0]) #plot each trace
plt.show()