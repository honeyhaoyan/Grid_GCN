#%load_ext line_profiler
#%%writefile simulation.py
#import numpy as np
#%load_ext line_profiler
#import line_profiler


import numpy as np


@profile
def step(*shape):
    # Create a random n-vector with +1 or -1 values.
    return 2 * (np.random.random_sample(shape)<.5) - 1

@profile
def simulate(iterations, n=10000):
    s = step(iterations, n)
    x = np.cumsum(s, axis=0)
    bins = np.arange(-30, 30, 1)
    y = np.vstack([np.histogram(x[i,:], bins)[0]
                   for i in range(iterations)])
    return y

simulate(50)

#from simulation import simulate

#%lprun -T lprof0 -f simulate simulate(50)
#*** Profile printout saved to text file 'lprof0'.

#print(open('lprof0', 'r').read())

