from __future__ import annotations

from typing import Dict, Iterable, Optional

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
    生成 vod_qc_array (uint8)。

    修改后的核心规则（支持“部分波段缺失仍保留其他波段”）：
      - qc_flag 仅由 QC(uint16) + extra_nan_mask 决定，不再把“任一VOD通道NaN”合并成全局无效。
      - 6: 填充值(255) 或 extra_nan_mask 为真
      - 5: snow/ice > 5%
      - 4: severe spectral RFI
      - 3: medium spectral RFI
      - 2: polarization RFI
      - 1: frozen
      - 0: good (qc_u8==0 且未触发上述规则)

    说明：
      - vod_dict 仍作为接口参数保留（保持调用处不变），但不再用于“跨通道NaN→全无效”的判定。
      - 各通道自身的 NaN 将在下游（extract / 写入 df）自然表现为该通道缺失，不牵连其它通道。
    """
    masks = decode_qc_masks(qc_uint16, fill_qc_value=fill_qc_value)
    qc_u8 = masks["qc_u8"]

    vod_qc = np.zeros_like(qc_u8, dtype=np.uint8)

    # 6: fill or extra_nan_mask
    vod_qc[masks["fill_mask"]] = 6
    if extra_nan_mask is not None:
        vod_qc[extra_nan_mask.astype(bool)] = 6

    # 仅在未被标记为6的像元上继续判别
    valid = (vod_qc != 6)

    # 5 snow/ice
    vod_qc[valid & masks["snow_ice_mask"]] = 5

    # 4 severe spectral RFI
    valid2 = valid & (vod_qc == 0)
    vod_qc[valid2 & masks["rfi_spec_severe"]] = 4

    # 3 medium spectral RFI
    valid3 = valid & (vod_qc == 0)
    vod_qc[valid3 & masks["rfi_spec_medium"]] = 3

    # 2 polarization RFI
    valid4 = valid & (vod_qc == 0)
    vod_qc[valid4 & masks["rfi_polar_mask"]] = 2

    # 1 frozen
    valid5 = valid & (vod_qc == 0)
    vod_qc[valid5 & masks["frozen_mask"]] = 1

    # 0 good: 严格要求 qc_u8==0（且未被上面标记）
    remain = (vod_qc == 0)
    vod_qc[remain & (qc_u8 != 0)] = 0  # 保持0不变；如需更严格，可改成6或其他flag

    return vod_qc


def build_valid_mask(vod_qc_array: np.ndarray, keep_flags: Iterable[int] = (0,)) -> np.ndarray:
    """根据 keep_flags 得到有效掩膜。"""
    keep_flags = np.array(list(keep_flags), dtype=np.uint8)
    return np.isin(vod_qc_array, keep_flags)


def apply_qc_mask_to_vod(
    vod_dict: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    fill_value: float = np.nan
) -> Dict[str, np.ndarray]:
    """
    将无效像元设置为 fill_value（默认 NaN），返回新的 vod_dict。

    注意：这里仍然是“QC无效→所有通道同步置NaN”的逻辑，
    但不会再因为“某个通道自身NaN”导致全通道被判无效；
    通道自身NaN会原样保留，从而实现“部分波段缺失但仍保留其他波段”。
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
