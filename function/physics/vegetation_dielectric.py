from __future__ import annotations

import numpy as np

# Vacuum permittivity (F/m)
EPS0 = 8.854e-12


def mg_from_lfmc(lfmc_pct: np.ndarray) -> np.ndarray:
    """
    LFMC(%) -> Mg in [0,1)
    LFMC = 100 * Mg/(1-Mg)  => Mg = LFMC/(LFMC+100)
    """
    lfmc = np.asarray(lfmc_pct, dtype=float)
    mg = lfmc / (lfmc + 100.0)
    return np.clip(mg, 0.0, 0.999999)


def lfmc_from_mg(mg: np.ndarray) -> np.ndarray:
    mg = np.asarray(mg, dtype=float)
    mg = np.clip(mg, 0.0, 0.999999)
    return 100.0 * mg / (1.0 - mg)


def sigma_from_salinity_permil(s_permil: np.ndarray) -> np.ndarray:
    """
    Approximation you provided:
      sigma ≈ 0.16 S - 0.0013 S^2   (S/m)
    where S is salinity in per mille (‰), intended for S <= ~10‰.

    Note: if you want sigma≈1.27 S/m, solve gives S≈8.53‰.
    """
    s = np.asarray(s_permil, dtype=float)
    sigma = 0.16 * s - 0.0013 * (s ** 2)
    return np.clip(sigma, 0.0, None)


def sigma_temp_correct(
    sigma_ref: np.ndarray,
    temp_c: np.ndarray,
    ref_c: float = 22.0,
    alpha_per_c: float = 0.02
) -> np.ndarray:
    """
    Simple conductivity temperature correction:
      sigma(T) = sigma(ref) * (1 + alpha*(T - ref))
    """
    sigma_ref = np.asarray(sigma_ref, dtype=float)
    t = np.asarray(temp_c, dtype=float)
    factor = 1.0 + alpha_per_c * (t - ref_c)
    factor = np.clip(factor, 0.0, None)
    return sigma_ref * factor


# -----------------------
# Free water dielectric: generalized Debye + conductivity
# -----------------------
def eps_s_water_malmberg_maryott(temp_c: np.ndarray) -> np.ndarray:
    """
    Static permittivity of water eps_s(T) (approx):
      eps_s(T) = 87.740 - 0.400081 T + 9.398e-4 T^2 - 1.410e-6 T^3
    T in Celsius.
    """
    T = np.asarray(temp_c, dtype=float)
    return 87.740 - 0.400081 * T + 9.398e-4 * T**2 - 1.410e-6 * T**3


def tau1_water_ellison(temp_c: np.ndarray) -> np.ndarray:
    """
    Relaxation time tau1(T) (Ellison-like form):
      tau1(T) = c1 * exp(d1/(T + tc))
    with:
      c1 = 1.382264e-13 s, d1 = 652.7648, tc = 133.1383
    """
    T = np.asarray(temp_c, dtype=float)
    c1 = 1.382264e-13
    d1 = 652.7648
    tc = 133.1383
    return c1 * np.exp(d1 / (T + tc))


def f0_water_ghz(temp_c: np.ndarray) -> np.ndarray:
    """
    Debye relaxation frequency:
      f0(T) = 1/(2*pi*tau1(T))
    Returned in GHz.
    """
    tau = tau1_water_ellison(temp_c)
    f0_hz = 1.0 / (2.0 * np.pi * tau)
    return f0_hz / 1e9


def epsilon_free_water_debye(
    f_ghz: np.ndarray,
    *,
    temp_c: np.ndarray | None = None,
    sigma_s_m: np.ndarray | float = 0.0,
    eps_inf: float = 4.9,
    eps_s: np.ndarray | float | None = None,
    f0_ghz_val: np.ndarray | float | None = None,
) -> np.ndarray:
    """
    Generalized free-water dielectric spectrum (complex), using convention:
      eps = eps' - j eps''  (loss => imag <= 0)

    eps_f(f,T) = eps_inf(T)
               + (eps_s(T)-eps_inf(T)) / (1 + j*f/f0(T))
               - j*sigma/(2*pi*eps0*f)
    """
    f = np.asarray(f_ghz, dtype=float)
    f = np.clip(f, 1e-6, None)

    sigma = np.asarray(sigma_s_m, dtype=float)

    if eps_s is None or f0_ghz_val is None:
        if temp_c is not None:
            T = np.asarray(temp_c, dtype=float)
            if eps_s is None:
                eps_s = eps_s_water_malmberg_maryott(T)
            if f0_ghz_val is None:
                f0_ghz_val = f0_water_ghz(T)
        else:
            # room approximation: 4.9 + 75/(1 + j*f/18)
            if eps_s is None:
                eps_s = eps_inf + 75.0
            if f0_ghz_val is None:
                f0_ghz_val = 18.0

    eps_s = np.asarray(eps_s, dtype=float)
    f0 = np.asarray(f0_ghz_val, dtype=float)
    f0 = np.clip(f0, 1e-6, None)

    j = 1j

    # Debye term
    debye = (eps_s - eps_inf) / (1.0 + j * (f / f0))

    # Conductivity term: -j * sigma/(2*pi*eps0*f_Hz)
    cond = -j * (sigma / (2.0 * np.pi * EPS0 * (f * 1e9)))

    return (eps_inf + debye + cond).astype(np.complex128)


def epsilon_bound_water_colecole(f_ghz: np.ndarray) -> np.ndarray:
    """
    Bound-water mixture dielectric (Cole–Cole-like):
      eps_b(f) = 2.9 + 55 / (1 + (j*f/0.18)^(0.5))
    Convention eps = eps' - j eps'' yields imag <= 0 naturally here.
    """
    f = np.asarray(f_ghz, dtype=float)
    f = np.clip(f, 1e-6, None)
    j = 1j
    return (2.9 + 55.0 / (1.0 + (j * f / 0.18) ** 0.5)).astype(np.complex128)


# -----------------------
# Vegetation dielectric (Ulaby 1987 style)
# -----------------------
def epsilon_veg_ulaby_dc(
    mg: np.ndarray,
    f_ghz: np.ndarray,
    sigma_s_m: np.ndarray,
    *,
    temp_c: np.ndarray | None = None,
    eps_inf: float = 4.9,
    eps_s: np.ndarray | float | None = None,
    f0_ghz_val: np.ndarray | float | None = None,
    clip_vfw_nonneg: bool = True,
    clip_loss_nonneg: bool = True,
) -> np.ndarray:
    """
    Ulaby-style bulk vegetation dielectric constant (complex), convention:
      eps = eps' - j eps''  (loss => imag <= 0)

    eps_veg = eps_r(Mg) + v_fw(Mg)*eps_free(f,T,sigma) + v_b(Mg)*eps_bound(f)

    Parameterizations:
      eps_r = 1.7 - 0.74*Mg + 6.16*Mg^2
      v_fw  = Mg*(0.55*Mg - 0.076)
      v_b   = 4.64*Mg^2 / (1 + 7.36*Mg^2)
    """
    mg = np.asarray(mg, dtype=float)
    f = np.asarray(f_ghz, dtype=float)
    sigma = np.asarray(sigma_s_m, dtype=float)

    mg = np.clip(mg, 0.0, 0.999999)
    f = np.clip(f, 1e-6, None)

    eps_r = 1.7 - 0.74 * mg + 6.16 * (mg ** 2)
    v_fw = mg * (0.55 * mg - 0.076)
    v_b = 4.64 * (mg ** 2) / (1.0 + 7.36 * (mg ** 2))

    if clip_vfw_nonneg:
        v_fw = np.clip(v_fw, 0.0, None)

    eps_free = epsilon_free_water_debye(
        f_ghz=f,
        temp_c=temp_c,
        sigma_s_m=sigma,
        eps_inf=eps_inf,
        eps_s=eps_s,
        f0_ghz_val=f0_ghz_val,
    )
    eps_bound = epsilon_bound_water_colecole(f)

    eps_v = eps_r + v_fw * eps_free + v_b * eps_bound

    if clip_loss_nonneg:
        # enforce eps'' >= 0  <=> imag(eps) <= 0 (since eps = eps' - j eps'')
        eps_v = eps_v.real + 1j * np.minimum(eps_v.imag, 0.0)

    return eps_v.astype(np.complex128)


def epsilon_veg_from_lfmc(
    lfmc_pct: np.ndarray,
    f_ghz: np.ndarray,
    model: str = "ulaby_dc",
    *,
    # --- temperature / salinity controls ---
    temp_c: np.ndarray | None = None,
    salinity_permil: np.ndarray | None = None,
    salinity_default_permil: float = 0.0,
    alpha_sigma_per_c: float = 0.02,
    sigma_ref_temp_c: float = 22.0,
    # --- free water model overrides (optional) ---
    eps_inf: float = 4.9,
    eps_s: np.ndarray | float | None = None,
    f0_ghz_val: np.ndarray | float | None = None,
    # --- flags ---
    clip_vfw_nonneg: bool = True,
    clip_loss_nonneg: bool = True,
    # --- placeholder fallback ---
    eps_dry_re: float = 1.6,
    eps_dry_im_loss: float = 0.05,  # interpreted as eps'' (loss)
    k_re: float = 25.0,
    k_im_loss: float = 10.0,
) -> np.ndarray:
    """
    Unified interface: LFMC(%) -> epsilon_veg (complex)

    - model="ulaby_dc": Ulaby-style mixing with generalized free-water Debye+conductivity.
    - model="rational": placeholder (ablation/debug), returns eps = eps' - j eps''

    Notes:
    - For model="ulaby_dc", if temp_c is provided, it affects sigma (via simple correction)
      and also (if you don't override eps_s/f0) affects eps_s(T) and f0(T) via formulas.
    """
    lfmc = np.asarray(lfmc_pct, dtype=float)
    f = np.asarray(f_ghz, dtype=float)

    if model.lower() == "ulaby_dc":
        mg = mg_from_lfmc(lfmc)

        # salinity -> sigma(ref_temp)
        if salinity_permil is None:
            s = np.full_like(mg, float(salinity_default_permil), dtype=float)
        else:
            s = np.asarray(salinity_permil, dtype=float)

        sigma_ref = sigma_from_salinity_permil(s)

        # temperature correction for conductivity
        if temp_c is None:
            sigma = sigma_ref
            temp_for_eps = None
        else:
            t = np.asarray(temp_c, dtype=float)
            sigma = sigma_temp_correct(
                sigma_ref=sigma_ref,
                temp_c=t,
                ref_c=float(sigma_ref_temp_c),
                alpha_per_c=float(alpha_sigma_per_c),
            )
            temp_for_eps = t

        return epsilon_veg_ulaby_dc(
            mg=mg,
            f_ghz=f,
            sigma_s_m=sigma,
            temp_c=temp_for_eps,
            eps_inf=eps_inf,
            eps_s=eps_s,
            f0_ghz_val=f0_ghz_val,
            clip_vfw_nonneg=clip_vfw_nonneg,
            clip_loss_nonneg=clip_loss_nonneg,
        )

    if model.lower() == "rational":
        # simple monotonic placeholder: eps = eps' - j eps'' (loss)
        m = lfmc / (100.0 + np.maximum(lfmc, 0.0))
        m = np.clip(m, 0.0, 0.999999)
        eps_re = eps_dry_re + k_re * m
        eps_im_loss = eps_dry_im_loss + k_im_loss * m
        return (eps_re - 1j * eps_im_loss).astype(np.complex128)

    raise ValueError(f"Unknown vegetation dielectric model: {model}")
