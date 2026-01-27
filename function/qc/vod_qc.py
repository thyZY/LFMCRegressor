from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def decode_qc_masks(qc_uint16: np.ndarray, fill_qc_value: int = 255) -> Dict[str, np.ndarray]:
    """
    将QC(uint16)解析成各mask。
    按你原脚本：只看低8位（uint8），并定义位含义：
      bit0 frozen
      bit1 RFI polarization
      bit2-3 spectral RFI (00 none, 01 medium, 10 severe)
      bit4 snow/ice >5%
      255 fill
    """
    qc_u8 = (qc_uint16 & 0xFF).astype(np.uint8)

    fill_mask = (qc_u8 == np.uint8(fill_qc_value))
    frozen_mask = (qc_u8 & 0b00000001) != 0
    rfi_polar_mask = (qc_u8 & 0b00000010) != 0
    spectral = (qc_u8 & 0b00001100) >> 2
    rfi_spec_medium = (spectral == 1)      # 01
    rfi_spec_severe = (spectral == 2)      # 10
    snow_ice_mask = (qc_u8 & 0b00010000) != 0

    return {
        "qc_u8": qc_u8,
        "fill_mask": fill_mask,
        "frozen_mask": frozen_mask,
        "rfi_polar_mask": rfi_polar_mask,
        "rfi_spec_medium": rfi_spec_medium,
        "rfi_spec_severe": rfi_spec_severe,
        "snow_ice_mask": snow_ice_mask,
    }


def build_vod_qc_array(
    qc_uint16: np.ndarray,
    vod_dict: Dict[str, np.ndarray],
    fill_qc_value: int = 255,
    extra_nan_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    生成 vod_qc_array (uint8)，规则与你原脚本一致：
      6: 填充值 或 任一输入变量 NaN
      5: snow/ice > 5%
      4: severe spectral RFI
      3: medium spectral RFI
      2: polarization RFI
      1: frozen
      0: good (qc==0 且无NaN)
    """
    masks = decode_qc_masks(qc_uint16, fill_qc_value=fill_qc_value)
    qc_u8 = masks["qc_u8"]

    # 输入变量NaN
    input_nan = np.zeros_like(qc_u8, dtype=bool)
    for k, arr in vod_dict.items():
        input_nan |= np.isnan(arr)

    if extra_nan_mask is not None:
        input_nan |= extra_nan_mask

    vod_qc = np.zeros_like(qc_u8, dtype=np.uint8)

    # 6: fill or NaN
    vod_qc[masks["fill_mask"]] = 6
    vod_qc[input_nan] = 6

    valid = (~masks["fill_mask"]) & (~input_nan)

    # 5 snow/ice
    vod_qc[valid & masks["snow_ice_mask"]] = 5

    # 4 severe
    valid2 = valid & (vod_qc == 0)
    vod_qc[valid2 & masks["rfi_spec_severe"]] = 4

    # 3 medium
    valid3 = (vod_qc == 0)
    vod_qc[valid3 & masks["rfi_spec_medium"]] = 3

    # 2 polar rfi
    valid4 = (vod_qc == 0)
    vod_qc[valid4 & masks["rfi_polar_mask"]] = 2

    # 1 frozen
    valid5 = (vod_qc == 0)
    vod_qc[valid5 & masks["frozen_mask"]] = 1

    # 0 good: qc==0 且无NaN 且没被上面标记
    # 这里 vod_qc==0 的地方包含 qc==0 也可能包含 qc_u8!=0 但未触发规则的情况
    # 为严格，强制 qc_u8==0 才算good；否则仍保留0或可改为6/2等
    remain = (vod_qc == 0)
    vod_qc[remain & (qc_u8 != 0)] = 0  # 维持0不变（若你想更严格可改为6）

    return vod_qc


def build_valid_mask(vod_qc_array: np.ndarray, keep_flags: Iterable[int] = (0,)) -> np.ndarray:
    """根据 keep_flags 得到有效掩膜。"""
    keep_flags = np.array(list(keep_flags), dtype=np.uint8)
    valid = np.isin(vod_qc_array, keep_flags)
    return valid


def apply_qc_mask_to_vod(
    vod_dict: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    fill_value: float = np.nan
) -> Dict[str, np.ndarray]:
    """
    将无效像元设置为 fill_value（默认 NaN），返回新的 vod_dict。
    """
    out = {}
    for k, arr in vod_dict.items():
        a = arr.copy()
        a[~valid_mask] = fill_value
        out[k] = a
    return out


def qc_counts(vod_qc_array: np.ndarray) -> Dict[int, int]:
    """返回各 qc flag 的像元个数统计。"""
    vals, counts = np.unique(vod_qc_array, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def qc_ratio(vod_qc_array: np.ndarray) -> Dict[int, float]:
    """返回各 qc flag 的比例统计。"""
    total = float(vod_qc_array.size)
    cnt = qc_counts(vod_qc_array)
    return {k: v / total for k, v in cnt.items()}
