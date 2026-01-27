import numpy as np
from scipy.optimize import root


# ---------------------------
# Helpers
# ---------------------------
def _normalize_fracs(fracs: np.ndarray) -> np.ndarray:
    fracs = np.asarray(fracs, dtype=float)
    s = np.sum(fracs)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("fracs must sum to a positive finite number.")
    return fracs / s


def _as_complex(x) -> np.ndarray:
    return np.asarray(x, dtype=np.complex128)


def _validate_depolarization(Aa: float, Ab: float, Ac: float, tol: float = 1e-8) -> None:
    s = Aa + Ab + Ac
    if not np.isfinite(s):
        raise ValueError("Aa, Ab, Ac must be finite.")
    if abs(s - 1.0) > tol:
        raise ValueError(f"Aa+Ab+Ac must equal 1 (got {s}).")
    if (Aa < 0) or (Ab < 0) or (Ac < 0):
        raise ValueError("Aa, Ab, Ac must be non-negative.")


# ---------------------------
# (1) Generalized Looyenga / Power-law mixing
# ---------------------------
def power_law_mix(eps_components: np.ndarray, fracs: np.ndarray, beta: float = 1.0 / 3.0) -> complex:
    """
    Generalized power-law (Looyenga/Birchak family) mixing:

        eps_eff^beta = sum_i f_i * eps_i^beta
        => eps_eff = (sum_i f_i * eps_i^beta)^(1/beta)

    - beta = 1/3 : Looyenga
    - beta = 1/2 : square-root / refractive index style
    - beta = 1   : linear mixing

    Works for complex eps (principal branch for complex powers).

    Parameters
    ----------
    eps_components : array-like of complex, shape (n_components,)
    fracs : array-like of float, same shape
        Volume fractions; will be normalized internally.
    beta : float
        Mixing exponent.

    Returns
    -------
    eps_eff : complex
    """
    if beta == 0:
        raise ValueError("beta cannot be 0 for power-law mixing.")
    eps_components = _as_complex(eps_components)
    fracs = _normalize_fracs(fracs)

    # principal complex power
    mixed = np.sum(fracs * np.power(eps_components, beta))
    return np.power(mixed, 1.0 / beta)


def power_law_air_veg(eps_veg: np.ndarray, vegf: np.ndarray, beta: float = 1.0 / 3.0) -> np.ndarray:
    """
    Special 2-phase case (air + vegetation) matching the form in your screenshot:

        eps_can = [ eps_air^beta + vegf * (eps_veg^beta - eps_air^beta) ]^(1/beta)

    With eps_air = 1:
        eps_can = [ 1 + vegf * (eps_veg^beta - 1) ]^(1/beta)

    Parameters
    ----------
    eps_veg : complex array-like
    vegf : float or array-like in [0,1]
    beta : float, default 1/3

    Returns
    -------
    eps_can : complex array-like
    """
    if beta == 0:
        raise ValueError("beta cannot be 0.")
    eps_veg = _as_complex(eps_veg)
    vegf = np.asarray(vegf, dtype=float)

    if np.any(vegf < 0) or np.any(vegf > 1):
        raise ValueError("vegf must be within [0, 1].")

    term = 1.0 + vegf * (np.power(eps_veg, beta) - 1.0)
    return np.power(term, 1.0 / beta)


# ---------------------------
# Existing: Bruggeman mixing
# ---------------------------
def bruggeman_mix(
    eps_components: np.ndarray,
    fracs: np.ndarray,
    init: complex = 1.5 + 0.01j
) -> complex:
    """
    Symmetric Bruggeman mixing for N components:
        sum_i f_i * (eps_i - eps_eff)/(eps_i + 2 eps_eff) = 0

    Solve complex root by converting to R^2.
    """
    eps_components = _as_complex(eps_components)
    fracs = _normalize_fracs(fracs)

    def F(xy):
        ee = xy[0] + 1j * xy[1]
        val = np.sum(fracs * (eps_components - ee) / (eps_components + 2.0 * ee))
        return np.array([val.real, val.imag], dtype=float)

    sol = root(F, np.array([init.real, init.imag], dtype=float), method="hybr")
    if not sol.success:
        # fallback: return init (do not crash)
        return init
    return sol.x[0] + 1j * sol.x[1]


# ---------------------------
# (2) Ulaby: Polder–van Santen / de Loor (inverse: eps_can -> eps_veg)
# ---------------------------
def de_loor_pvs_forward(
    eps_host: complex,
    eps_incl: complex,
    vegf: float,
    Aa: float = 0.0,
    Ab: float = 0.0,
    Ac: float = 1.0,
    eps_star: complex | None = None,
) -> complex:
    """
    Forward Polder–van Santen / de Loor formula (Ulaby Eq. 4.27 style):

        eps_m = eps_h + (v_i/3) * (eps_i - eps_h) * sum_{u=a,b,c} [ 1 / (1 + A_u*(eps_i/eps* - 1)) ]

    where:
      eps_h : host dielectric (air=1 for canopy air)
      eps_i : inclusion dielectric (vegetation)
      v_i   : inclusion volume fraction (vegf)
      A_u   : depolarization factors (Aa, Ab, Ac), sum to 1
      eps*  : effective dielectric around each particle:
              - for small v_i, often eps* ≈ eps_h
              - for higher v_i, can use eps* ≈ eps_m

    This function lets you pass eps_star explicitly (recommended for clarity).

    Returns
    -------
    eps_m : complex
        Mixture dielectric (canopy equivalent dielectric).
    """
    _validate_depolarization(Aa, Ab, Ac)
    v = float(vegf)
    if v < 0 or v > 1:
        raise ValueError("vegf must be within [0,1].")

    eps_h = complex(eps_host)
    eps_i = complex(eps_incl)
    eps_s = eps_h if eps_star is None else complex(eps_star)

    # Avoid division by zero in eps_i/eps_s
    if eps_s == 0:
        raise ValueError("eps_star cannot be 0.")

    ratio_term = (eps_i / eps_s) - 1.0
    summation = (
        1.0 / (1.0 + Aa * ratio_term)
        + 1.0 / (1.0 + Ab * ratio_term)
        + 1.0 / (1.0 + Ac * ratio_term)
    )

    eps_m = eps_h + (v / 3.0) * (eps_i - eps_h) * summation
    return eps_m


def de_loor_pvs_invert_eps_veg(
    eps_canopy: complex,
    vegf: float,
    Aa: float = 0.0,
    Ab: float = 0.0,
    Ac: float = 1.0,
    *,
    eps_air: complex = 1.0 + 0.0j,
    v_threshold: float = 0.1,
    init: complex = 10.0 + 1.0j,
    method: str = "hybr",
) -> complex:
    """
    Invert Ulaby Polder–van Santen / de Loor formula to solve eps_veg (inclusion dielectric),
    given canopy mixture dielectric eps_canopy and vegf.

    eps* rule (as you requested: changes with vegf):
      - if vegf <= v_threshold: eps* = eps_host (air)
      - else:                  eps* = eps_m (mixture) = eps_canopy

    Inputs (as you asked)
    ---------------------
    eps_canopy : complex
        Canopy equivalent dielectric (mixture).
    vegf : float in [0,1]
        Vegetation volume fraction.
    Aa, Ab, Ac : float
        Depolarization factors, default (0,0,1). Must satisfy Aa+Ab+Ac=1.

    Returns
    -------
    eps_veg : complex
        Vegetation dielectric constant (inclusion).
    """
    _validate_depolarization(Aa, Ab, Ac)
    v = float(vegf)
    if v < 0 or v > 1:
        raise ValueError("vegf must be within [0,1].")

    eps_m_target = complex(eps_canopy)
    eps_h = complex(eps_air)

    # Choose eps_star based on vegf
    if v <= v_threshold:
        eps_star = eps_h
    else:
        eps_star = eps_m_target

    def F(xy):
        eps_i = xy[0] + 1j * xy[1]
        eps_m = de_loor_pvs_forward(
            eps_host=eps_h,
            eps_incl=eps_i,
            vegf=v,
            Aa=Aa, Ab=Ab, Ac=Ac,
            eps_star=eps_star,
        )
        diff = eps_m - eps_m_target
        return np.array([diff.real, diff.imag], dtype=float)

    sol = root(F, np.array([init.real, init.imag], dtype=float), method=method)
    if not sol.success:
        # fallback: return init (do not crash)
        return init
    return sol.x[0] + 1j * sol.x[1]
