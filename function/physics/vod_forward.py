import numpy as np

def vod_from_eps_canopy(eps_canopy, wavelength_m, hveg_m, a, b):
    eps_canopy = np.asarray(eps_canopy, dtype=np.complex128)
    hveg_m = np.asarray(hveg_m, dtype=float)

    sqrt_eps = np.sqrt(eps_canopy)
    im_term = np.abs(np.imag(sqrt_eps))  # force non-negative

    scale = 4.0 * np.pi * (b * hveg_m + a) / float(wavelength_m)
    return scale * im_term
