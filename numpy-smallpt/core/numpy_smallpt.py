import numpy as np

from image_io import write_ppm
from math_tools import normalize
from ray import Ray
from rng import RNG
from sampling import cosine_weighted_sample_on_hemisphere
from sphere import Sphere
from specular import ideal_specular_reflect, ideal_specular_transmit

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5

spheres = [
        Sphere(1e5,  np.array([1e5 + 1, 40.8, 81.6],    dtype=np.float64), f=np.array([0.75,0.25,0.25],      dtype=np.float64)),
	    Sphere(1e5,  np.array([-1e5 + 99, 40.8, 81.6],  dtype=np.float64), f=np.array([0.25,0.25,0.75],      dtype=np.float64)),
	    Sphere(1e5,  np.array([50, 40.8, 1e5],          dtype=np.float64), f=np.array([0.75, 0.75, 0.75],    dtype=np.float64)),
	    Sphere(1e5,  np.array([50, 40.8, -1e5 + 170],   dtype=np.float64)),
	    Sphere(1e5,  np.array([50, 1e5, 81.6],          dtype=np.float64), f=np.array([0.75, 0.75, 0.75],    dtype=np.float64)),
	    Sphere(1e5,  np.array([50, -1e5 + 81.6, 81.6],  dtype=np.float64), f=np.array([0.75, 0.75, 0.75],    dtype=np.float64)),
	    Sphere(16.5, np.array([27, 16.5, 47],           dtype=np.float64), f=np.array([0.999, 0.999, 0.999], dtype=np.float64), reflection_t=Sphere.Reflection_t.SPECULAR),
	    Sphere(16.5, np.array([73, 16.5, 78],           dtype=np.float64), f=np.array([0.999, 0.999, 0.999], dtype=np.float64), reflection_t=Sphere.Reflection_t.REFRACTIVE),
	    Sphere(600,  np.array([50, 681.6 - .27, 81.6],  dtype=np.float64), e=np.array([12, 12, 12],          dtype=np.float64))
        ]


def intersect(ray):
    id = None
    hit = False
    for i in range(len(spheres)):
        if spheres[i].intersect(ray):
            hit = True
            id = i
    return hit, id

def intersectP(ray):
    for i in range(len(spheres)):
        if spheres[i].intersect(ray):
            return True
    return False

def radiance(ray, rng):
    r = ray
    L = np.zeros((3), dtype=np.float64)
    F = np.ones((3), dtype=np.float64)

    while (True):
        hit, id = intersect(r)
        if (not hit):
            return L

        shape = spheres[id]
        p = r(r.tmax)
        n = normalize(p - shape.p)

        L += F * shape.e
        F *= shape.f
        
	    # Russian roulette
        if r.depth > 4:
            continue_probability = np.amax(shape.f)
            if rng.uniform_float() >= continue_probability:
                return L
            F /= continue_probability

        # Next path segment
        if shape.reflection_t == Sphere.Reflection_t.SPECULAR:
            d = ideal_specular_reflect(r.d, n)
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue
        elif shape.reflection_t == Sphere.Reflection_t.REFRACTIVE:
            d, pr = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, rng)
            F *= pr
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue
        else:
            w = n if n.dot(r.d) < 0 else -n
            u = normalize(np.cross(np.array([0.0, 1.0, 0.0], np.float64) if np.fabs(w[0]) > 0.1 else np.array([1.0, 0.0, 0.0], np.float64), w))
            v = np.cross(w, u)

            sample_d = cosine_weighted_sample_on_hemisphere(rng.uniform_float(), rng.uniform_float())
            d = normalize(sample_d[0] * u + sample_d[1] * v + sample_d[2] * w)
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue

import sys

if __name__ == "__main__":
    rng = RNG()
    nb_samples = int(sys.argv[1]) // 4 if len(sys.argv) > 1 else 1

    w = 1024
    h = 768

    eye = np.array([50, 52, 295.6], dtype=np.float64)
    gaze = normalize(np.array([0, -0.042612, -1], dtype=np.float64))
    fov = 0.5135
    cx = np.array([w * fov / h, 0.0, 0.0], dtype=np.float64)
    cy = normalize(np.cross(cx, gaze)) * fov

    Ls = np.zeros((w * h, 3), dtype=np.float64)
    
    for y in range(h):
        # pixel row
        print('\rRendering ({0} spp) {1:0.2f}%'.format(nb_samples * 4, 100.0 * y / (h - 1)))
        for x in range(w):
            # pixel column
            for sy in range(2):
                i = (h - 1 - y) * w + x
                # 2 subpixel row
                for sx in range(2):
                    # 2 subpixel column
                    L = np.zeros((3), dtype=np.float64)
                    for s in range(nb_samples):
                        #  samples per subpixel
                        u1 = 2.0 * rng.uniform_float()
                        u2 = 2.0 * rng.uniform_float()
                        dx = np.sqrt(u1) - 1.0 if u1 < 1 else 1.0 - np.sqrt(2.0 - u1)
                        dy = np.sqrt(u2) - 1.0 if u2 < 1 else 1.0 - np.sqrt(2.0 - u2)
                        d = cx * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + \
                            cy * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + gaze
                        L += radiance(Ray(eye + d * 130, normalize(d), tmin=Sphere.EPSILON_SPHERE), rng) * (1.0 / nb_samples)
                    Ls[i,:] += 0.25 * np.clip(L, a_min=0.0, a_max=1.0)

    write_ppm(w, h, Ls)
