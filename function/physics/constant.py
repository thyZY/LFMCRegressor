import numpy as np

C0 = 299_792_458.0  # speed of light (m/s)

def wavelength_from_freq(freq_hz: float) -> float:
    """lambda = c / f"""
    return C0 / float(freq_hz)
