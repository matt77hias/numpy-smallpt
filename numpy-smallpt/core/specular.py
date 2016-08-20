from math_tools import normalize
import numpy as np

def reflectance0(n1, n2):
    sqrt_R0 = np.float64(n1 - n2) / (n1 + n2)
    return sqrt_R0 * sqrt_R0

def schlick_reflectance(n1, n2, c):
    R0 = reflectance0(n1, n2)
    return R0 + (1.0 - R0) * c * c * c * c * c

def ideal_specular_reflect(d, n):
    return d - 2.0 * n.dot(d) * n

def ideal_specular_transmit(d, n, n_out, n_in, rng):
    n_out, n_in = np.float64(n_out), np.float64(n_in)
    d_Re = ideal_specular_reflect(d, n)

    out_to_in = n.dot(d) < 0
    nl = n if out_to_in else -n
    nn = n_out / n_in if out_to_in else n_in / n_out
    cos_theta = d.dot(nl)
    cos2_phi = 1.0 - nn * nn * (1.0 - cos_theta * cos_theta)

    # Total Internal Reflection
    if cos2_phi < 0:
        return d_Re, 1.0

    d_Tr = normalize(nn * d - nl * (nn * cos_theta + np.sqrt(cos2_phi)))
    c = 1.0 - (-cos_theta if out_to_in else d_Tr.dot(n))

    Re = schlick_reflectance(n_out, n_in, c)
    p_Re = 0.25 + 0.5 * Re
    if rng.uniform_float() < p_Re:
        return d_Re, (Re / p_Re)
    else:
        Tr = 1.0 - Re
        p_Tr = 1.0 - p_Re
        return d_Tr, (Tr / p_Tr)