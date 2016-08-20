import numpy as np

def uniform_sample_on_hemisphere(u1, u2):
    r = np.sqrt(np.max(0.0, 1.0 - u1 * u1))
    phi = 2.0 * np.pi * u2
    return np.array([r * np.cos(phi), r * np.sin(phi), u1], dtype=np.float64)