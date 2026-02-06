import numpy as np

def vod_from_eps_canopy(
    eps_canopy,
    wavelength_m,
    hveg_m,
    *,
    mode: str = "physics",          # "physics" | "proportional"
    eta: float | np.ndarray = 1.0,  # 物理尺度缩放（默认1，不改变原量级）
    alpha: float | np.ndarray | None = None,  # proportional 模式的系数
    clamp_max: float | None = None, # 可选：限制最大值防爆炸
):
    """
    1) mode="physics"（添加校正项 eta）：
        VOD = eta * [4*pi*hveg/lambda] * abs(Im(sqrt(eps_canopy)))

    2) mode="proportional"（正比形式）：
        VOD = alpha * abs(Im(sqrt(eps_canopy)))
    """
    eps_canopy = np.asarray(eps_canopy, dtype=np.complex128)
    hveg_m = np.asarray(hveg_m, dtype=float)

    sqrt_eps = np.sqrt(eps_canopy)
    im_term = -np.imag(sqrt_eps)
    # im_term = np.abs(np.imag(sqrt_eps))  # 保持你现在的“正值虚部项”

    if mode == "physics":
        scale = 4.0 * np.pi * hveg_m / float(wavelength_m)
        vod = np.asarray(eta, dtype=float) * scale * im_term

    elif mode == "proportional":
        if alpha is None:
            raise ValueError("mode='proportional' requires alpha")
        vod = np.asarray(alpha, dtype=float) * im_term

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if clamp_max is not None:
        vod = np.clip(vod, 0.0, float(clamp_max))

    return vod
