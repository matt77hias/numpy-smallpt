import numpy as np

def uniform_sample_on_hemisphere(u1, u2):
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - u1 * u1))
    phi = 2.0 * np.pi * u2
    return np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, u1], dtype=np.float64)
	
def cosine_weighted_sample_on_hemisphere(u1, u2):
    cos_theta = np.sqrt(1.0 - u1)
    sin_theta = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    return np.array([np.cos(phi) * sin_theta, np.sin(phi) * sin_theta, cos_theta], dtype=np.float64)