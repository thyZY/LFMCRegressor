from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)

_DATE8_RE = re.compile(r"(\d{8})")

def scan_available_dates(dir_path: str) -> List[datetime]:
    """
    Scan a directory and extract all YYYYMMDD dates from filenames.
    Only files containing an 8-digit date are included.
    Returns sorted unique datetime list (date only).
    """
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(dir_path)

    dates: List[datetime] = []
    for fn in os.listdir(dir_path):
        m = _DATE8_RE.search(fn)
        if not m:
            continue
        s = m.group(1)
        try:
            d = datetime.strptime(s, "%Y%m%d")
            dates.append(d)
        except Exception:
            continue

    dates = sorted(set(dates))
    if not dates:
        logger.warning(f"No dated files found in: {dir_path}")
    return dates


def choose_prev_date(target: datetime, available: List[datetime]) -> Optional[datetime]:
    """
    Choose the 'prev' (window ownership) date:
      choose max(d in available) such that d <= target
    If none exists, return the earliest available (or None if available empty).
    """
    if not available:
        return None

    # binary search manually (available sorted)
    lo, hi = 0, len(available) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        d = available[mid]
        if d <= target:
            best = d
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        # target earlier than earliest
        best = available[0]
        logger.warning(
            f"Target date {target.strftime('%Y-%m-%d')} earlier than earliest available "
            f"{best.strftime('%Y-%m-%d')}; using earliest."
        )
    return best


def build_file_path_by_date(dir_path: str, date: datetime, suffix: str = ".h5") -> str:
    """
    Build 'YYYYMMDD{suffix}' full path.
    """
    return os.path.join(dir_path, date.strftime("%Y%m%d") + suffix)
