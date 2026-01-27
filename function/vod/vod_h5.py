# function/vod_h5.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import h5py


def find_vod_file(date: datetime, vod_cfg: dict) -> Optional[str]:
    """
    根据日期在 base_path 下查找可能的 VOD h5 文件。
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
    统一为 (rows, cols)；若为 (cols, rows) 则转置。
    """
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D array, got shape={arr.shape}")

    if arr.shape == (rows, cols):
        return arr
    if arr.shape == (cols, rows):
        return arr.T
    raise ValueError(f"Unsupported grid shape {arr.shape}, expected ({rows},{cols}) or ({cols},{rows})")


def _read_dataset_anywhere(f: h5py.File, key: str) -> np.ndarray:
    """
    在 h5 中读取 dataset：
    - 先尝试根目录直接 key、key.lower()、key.upper()
    - 若找不到，遍历所有 dataset，匹配“最后一段名字”等于 key（忽略大小写）
    - key 也可以是完整路径（如 'group1/ku_vod_H'）
    """
    # 1) key 是路径的情况
    if "/" in key:
        if key in f:
            return np.array(f[key][:])
        # 允许开头带 /
        key2 = key.lstrip("/")
        if key2 in f:
            return np.array(f[key2][:])

    # 2) 根目录直接匹配
    for k in (key, key.lower(), key.upper()):
        if k in f:
            return np.array(f[k][:])

    # 3) 遍历匹配 dataset 的 basename
    key_lower = key.lower()
    hit_path = None

    def _visitor(name, obj):
        nonlocal hit_path
        if hit_path is not None:
            return
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1].lower()
            if base == key_lower:
                hit_path = name

    f.visititems(_visitor)

    if hit_path is None:
        raise KeyError(f"Dataset '{key}' not found (also searched by basename)")

    return np.array(f[hit_path][:])


def read_vod_h5(file_path: str, vod_cfg: dict) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    读取单个 h5 文件，返回：
      vod_dict: {std_name: (1800,3600)} float32
      qc_uint16: (1800,3600) uint16
    """
    rows = int(vod_cfg["grid"]["rows"])
    cols = int(vod_cfg["grid"]["cols"])
    var_map = vod_cfg["var_map"]          # {h5_var: std_name}
    qc_var = vod_cfg.get("qc_var", "QC")  # QC dataset name

    vod_dict: Dict[str, np.ndarray] = {}

    def _print_ds_info(f: h5py.File, key: str):
        """
        尽量把 dataset 的压缩/filter/chunk 信息打印出来，帮助判断 filter failure 根因。
        注意：不改变读取逻辑，仅用于 debug 打印。
        """
        # 1) 先用最常见的直接命中方式找 ds
        for k in (key, key.lower(), key.upper()):
            if k in f:
                ds = f[k]
                print("  [DS FOUND @ root] name:", k, "path:", ds.name)
                print("  shape:", ds.shape, "dtype:", ds.dtype, "ndim:", ds.ndim)
                print("  chunks:", ds.chunks)
                print("  compression:", ds.compression, "compression_opts:", ds.compression_opts)
                if hasattr(ds, "filters"):
                    print("  filters:", ds.filters)
                else:
                    print("  filters: <not available in this h5py version>")
                return

        # 2) 尝试在全树里按 basename 匹配（与你 _read_dataset_anywhere 的思路一致）
        key_lower = key.lower()
        hit_paths = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                base = name.split("/")[-1].lower()
                if base == key_lower:
                    hit_paths.append("/" + name)

        try:
            f.visititems(visitor)
        except Exception as e:
            print("  [WARN] f.visititems failed:", repr(e))

        if hit_paths:
            ds = f[hit_paths[0]]
            print("  [DS FOUND @ tree] key:", key, "hit:", hit_paths[0])
            print("  shape:", ds.shape, "dtype:", ds.dtype, "ndim:", ds.ndim)
            print("  chunks:", ds.chunks)
            print("  compression:", ds.compression, "compression_opts:", ds.compression_opts)
            if hasattr(ds, "filters"):
                print("  filters:", ds.filters)
            else:
                print("  filters: <not available in this h5py version>")
        else:
            print("  [DS NOT FOUND FOR DEBUG] key:", key)
            print("  top keys:", list(f.keys())[:50])

    with h5py.File(file_path, "r") as f:
        # --- VOD vars ---
        for h5_var, std_key in var_map.items():
            try:
                arr = _read_dataset_anywhere(f, h5_var)
            except OSError as e:
                print("\n!!! OSError while reading VOD var !!!")
                print("file:", file_path)
                print("h5_var:", h5_var, "-> std_key:", std_key)
                print("error:", repr(e))
                print("[Dataset debug info]")
                _print_ds_info(f, h5_var)
                raise

            arr = np.squeeze(arr)  # 防止 (1,1800,3600) 这种
            if arr.ndim != 2:
                raise ValueError(f"{h5_var} after squeeze still not 2D: shape={arr.shape}")

            arr = _ensure_grid_shape(arr, rows, cols).astype(np.float32)
            vod_dict[std_key] = arr

        # --- QC ---
        try:
            qc = _read_dataset_anywhere(f, qc_var)
        except OSError as e:
            print("\n!!! OSError while reading QC !!!")
            print("file:", file_path)
            print("qc_var:", qc_var)
            print("error:", repr(e))
            print("[Dataset debug info]")
            _print_ds_info(f, qc_var)
            raise

        qc = np.squeeze(qc)
        if qc.ndim != 2:
            raise ValueError(f"{qc_var} after squeeze still not 2D: shape={qc.shape}")

        qc = _ensure_grid_shape(qc, rows, cols).astype(np.uint16)

    return vod_dict, qc
