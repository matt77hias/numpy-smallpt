import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def to_byte(x, gamma = 2.2):
    return int(np.clip(255.0 * np.power(x, 1.0 / gamma), a_min=0.0, a_max=255.0))