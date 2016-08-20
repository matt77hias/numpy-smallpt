import numpy as np

class Ray(object):

    def __init__(self, o, d, tmin = 0.0, tmax = np.inf, depth = 0):
        self.o = np.copy(o)
        self.d = np.copy(d)
        self.tmin = tmin
        self.tmax = tmax
        self.depth = depth

    def __call__(self, t):
        return self.o + self.d * t

    def __str__(self):
        return 'o: ' + str(self.o) + '\n' + 'd: ' + str(self.d) + '\n'