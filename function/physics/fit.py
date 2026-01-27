from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .constants import wavelength_from_freq
from .vegetation_dielectric import epsilon_veg_from_lfmc
from .mixing import looyenga_mix, bruggeman_mix
from .vod_forward import vod_from_eps_canopy
from .parameterization import PhysicsParams, params_to_vector, vector_to_params, build_bounds


def _mix_canopy_eps(eps_air: complex, eps_veg: complex, delta: float, model: str) -> complex:
    """
    Mix canopy dielectric constant from air + vegetation using a selected EMT model.
    """
    fracs = np.array([1.0 - delta, delta], dtype=float)
    eps = np.array([eps_air, eps_veg], dtype=complex)

    if model == "looyenga":
        return looyenga_mix(eps, fracs)

    if model == "bruggeman":
        return bruggeman_mix(eps, fracs)

    raise ValueError(f"Unknown mixing model: {model}")


def predict_vod_dataframe(df: pd.DataFrame, cfg: Dict[str, Any], p: PhysicsParams) -> np.ndarray:
    """
    Predict VOD from LFMC using a physical chain:
      LFMC -> Mg -> epsilon_veg(f, salinity) -> epsilon_canopy -> VOD

    IMPORTANT (Plan A):
      - LST (temperature) is NOT used in the dielectric model.
      - You may keep lst_k/lst_c columns in df for later analyses, but they do not enter here.

    Required df columns:
      - lfmc_pct (float, %)
      - hveg_m   (float, m)
      - band     (str: C/X/Ku)

    Optional df columns:
      - salinity_permil (float, ‰)  # if absent, uses default in config

    Returns:
      - vod_pred (float array)
    """
    # --- configs ---
    bands_hz = cfg["bands_hz"]
    mix_cfg = cfg["canopy_mixing"]
    veg_cfg = cfg["vegetation_dielectric"]

    # --- inputs ---
    lfmc = df["lfmc_pct"].to_numpy(dtype=float)
    hveg = df["hveg_m"].to_numpy(dtype=float)
    band = df["band"].astype(str).to_numpy()

    # per-row frequency in GHz
    f_ghz_vec = np.array([float(bands_hz[b]) / 1e9 for b in band], dtype=float)

    # salinity vector (‰) if provided; else use default
    if "salinity_permil" in df.columns:
        salinity_vec = df["salinity_permil"].to_numpy(dtype=float)
    else:
        salinity_vec = None

    # eps_air
    eps_air = 1.0 + 0.0j

    # epsilon_veg: Ulaby DC (temperature not used)
    eps_veg_vec = epsilon_veg_from_lfmc(
        lfmc_pct=lfmc,
        f_ghz=f_ghz_vec,
        model=str(veg_cfg.get("model", "ulaby_dc")),
        temp_c=None,  # <- Plan A: do NOT use LST/temperature in physics
        salinity_permil=salinity_vec,
        salinity_default_permil=float(veg_cfg.get("salinity_default_permil", 0.0)),
        clip_vfw_nonneg=bool(veg_cfg.get("clip_vfw_nonneg", True)),
        clip_im_nonneg=bool(veg_cfg.get("clip_im_nonneg", True)),
        # fallback params (only used if model == "rational")
        eps_dry_re=float(veg_cfg.get("params", {}).get("eps_dry_re", 1.6)),
        eps_dry_im=float(veg_cfg.get("params", {}).get("eps_dry_im", 0.05)),
        k_re=float(veg_cfg.get("params", {}).get("k_re", 25.0)),
        k_im=float(veg_cfg.get("params", {}).get("k_im", 10.0)),
    )

    # --- forward VOD ---
    vod_pred = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        b = band[i]
        freq = float(bands_hz[b])
        lam = wavelength_from_freq(freq)

        a = getattr(p, f"a_{b}")
        bb = getattr(p, f"b_{b}")

        eps_canopy = _mix_canopy_eps(
            eps_air=eps_air,
            eps_veg=eps_veg_vec[i],
            delta=float(p.delta),
            model=str(mix_cfg.get("model", "looyenga")),
        )

        vod_pred[i] = float(vod_from_eps_canopy(eps_canopy, lam, hveg[i], a, bb))

    return vod_pred


def fit_physics_model(df: pd.DataFrame, cfg_all: Dict[str, Any]) -> Tuple[PhysicsParams, Dict[str, Any]]:
    """
    Fit minimal parameter set using robust least squares.

    Required df columns:
      - lfmc_pct, hveg_m, band, vod_obs

    Returns:
      (best_params, fit_report)
    """
    cfg = cfg_all["physics"]
    fit_cfg = cfg["fit"]
    names = fit_cfg["fit_params"]

    # init from config
    cal = cfg["vod_calibration"]
    delta0 = float(cfg["canopy_mixing"]["delta_default"])

    # Note: k_re/k_im kept for compatibility if you still have them in PhysicsParams;
    #       for ulaby_dc, they are not used unless you switch model to "rational".
    veg_params = cfg.get("vegetation_dielectric", {}).get("params", {})

    p0 = PhysicsParams(
        a_C=float(cal["C"]["a"]), b_C=float(cal["C"]["b"]),
        a_X=float(cal["X"]["a"]), b_X=float(cal["X"]["b"]),
        a_Ku=float(cal["Ku"]["a"]), b_Ku=float(cal["Ku"]["b"]),
        delta=delta0,
        k_re=float(veg_params.get("k_re", 25.0)),
        k_im=float(veg_params.get("k_im", 10.0)),
    )

    x0 = params_to_vector(p0, names)
    lo, hi = build_bounds(names, fit_cfg["bounds"])

    y_obs = df["vod_obs"].to_numpy(dtype=float)

    def residuals(x):
        p = vector_to_params(x, p0, names)
        y_pred = predict_vod_dataframe(df, cfg, p)
        r = y_pred - y_obs
        # mask non-finite
        r = np.where(np.isfinite(r), r, 0.0)
        return r

    res = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        loss=fit_cfg.get("robust_loss", "linear"),
        f_scale=float(fit_cfg.get("f_scale", 1.0)),
        max_nfev=int(fit_cfg.get("max_nfev", 200)) if isinstance(fit_cfg.get("max_nfev", 200), (int, float)) else 200,
    )

    p_best = vector_to_params(res.x, p0, names)

    report = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "params": {n: float(getattr(p_best, n)) for n in names},
    }
    return p_best, report
