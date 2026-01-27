from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import h5py


def find_vod_file(date: datetime, vod_cfg: dict) -> Optional[str]:
    """
    根据日期在 base_path 下查找可能的 VOD mat 文件。
    date: datetime
    vod_cfg: configs/paths.yaml 里 vod 节点
    """
    base_path = vod_cfg["base_path"]
    templates = vod_cfg["filename_templates"]
    date_str = date.strftime("%Y%m%d")

    for tmpl in templates:
        fp = os.path.join(base_path, tmpl.format(date=date_str))
        if os.path.exists(fp):
            return fp
    return None


def _ensure_grid_shape(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    统一为 (rows, cols)。
    你的数据常见是 (3600,1800) 需要转置为 (1800,3600)。
    """
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D array, got shape={arr.shape}")

    if arr.shape == (rows, cols):
        return arr
    if arr.shape == (cols, rows):
        return arr.T

    # 有些mat可能额外维度被挤压，可自行扩展处理
    raise ValueError(f"Unsupported grid shape {arr.shape}, expected ({rows},{cols}) or ({cols},{rows})")


def read_vod_mat(file_path: str, vod_cfg: dict) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    读取单个 mat 文件（HDF5格式），返回：
      vod_dict: {"tau_Ku_H": (1800,3600), ...} float32
      qc_array: (1800,3600) uint16 (或转换后)
    """
    rows = int(vod_cfg["grid"]["rows"])
    cols = int(vod_cfg["grid"]["cols"])
    var_map = vod_cfg["var_map"]
    qc_var = vod_cfg.get("qc_var", "QC")

    vod_dict: Dict[str, np.ndarray] = {}

    with h5py.File(file_path, "r") as f:
        # 1) 读取VOD变量
        for mat_var, std_key in var_map.items():
            if mat_var in f:
                arr = np.array(f[mat_var][:])
            elif mat_var.lower() in f:
                arr = np.array(f[mat_var.lower()][:])
            elif mat_var.upper() in f:
                arr = np.array(f[mat_var.upper()][:])
            else:
                raise KeyError(f"Missing VOD variable '{mat_var}' in {os.path.basename(file_path)}")

            arr = _ensure_grid_shape(arr, rows, cols).astype(np.float32)
            vod_dict[std_key] = arr

        # 2) 读取QC变量
        if qc_var not in f and qc_var.lower() in f:
            qc_var = qc_var.lower()
        if qc_var not in f and qc_var.upper() in f:
            qc_var = qc_var.upper()
        if qc_var not in f:
            raise KeyError(f"Missing QC variable '{vod_cfg.get('qc_var','QC')}' in {os.path.basename(file_path)}")

        qc = np.array(f[qc_var][:])
        qc = _ensure_grid_shape(qc, rows, cols)

        # QC 常见是 uint16，但你规则只用低8位；这里先保留原始 uint16
        qc = qc.astype(np.uint16)

    return vod_dict, qc
