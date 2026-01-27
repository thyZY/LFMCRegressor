from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np

from .paths import AuxPaths
from .canopy_height import load_canopy_height
from .agb import load_agb_static_2015
from .igbp import load_igbp_year, find_available_igbp_years, choose_igbp_year, IGBP_CLASSES
from .glass_lai import load_lai_by_index
from .glass_fvc import load_fvc_by_index
from .lst import load_lst_daily

@dataclass
class AuxData:
    hveg: np.ndarray                      # (1800,3600)
    agb: np.ndarray                       # (1800,3600)
    igbp: Dict[str, np.ndarray]           # each (1800,3600)
    lai: np.ndarray                       # (1800,3600)
    fvc: np.ndarray                       # (1800,3600)
    lst_k: np.ndarray                     # (1800,3600)


class AuxDataLoader:
    """
    Loader with caching:
      - Hveg: static
      - AGB: static (2015)
      - IGBP: yearly
      - LAI/FVC/LST: daily
    """
    def __init__(self, paths: AuxPaths, igbp_classes=IGBP_CLASSES):
        self.paths = paths
        self.igbp_classes = list(igbp_classes)

        # caches
        self._hveg: Optional[np.ndarray] = None
        self._agb: Optional[np.ndarray] = None
        self._igbp_year_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._lai_cache: Dict[str, np.ndarray] = {}
        self._fvc_cache: Dict[str, np.ndarray] = {}
        self._lst_cache: Dict[str, np.ndarray] = {}

        # available IGBP years
        self._igbp_years = find_available_igbp_years(self.paths.igbp_dir)

    def load_hveg(self) -> np.ndarray:
        if self._hveg is None:
            self._hveg = load_canopy_height(self.paths.canopy_height_h5, "Hveg")
        return self._hveg

    def load_agb(self) -> np.ndarray:
        if self._agb is None:
            self._agb = load_agb_static_2015(self.paths.agb_dir, "AGB")
        return self._agb

    def load_igbp(self, year: int) -> Dict[str, np.ndarray]:
        # choose best available year (<= target year preferred)
        use_year = choose_igbp_year(year, self._igbp_years) if self._igbp_years else year
        if use_year not in self._igbp_year_cache:
            self._igbp_year_cache[use_year] = load_igbp_year(self.paths.igbp_dir, use_year, self.igbp_classes)
        return self._igbp_year_cache[use_year]

    def load_lai(self, date: datetime) -> np.ndarray:
        key = date.strftime("%Y%m%d")
        if key not in self._lai_cache:
            self._lai_cache[key] = load_lai_by_index(self.paths.lai_dir, date, "LAI")
        return self._lai_cache[key]

    def load_fvc(self, date: datetime) -> np.ndarray:
        key = date.strftime("%Y%m%d")
        if key not in self._fvc_cache:
            self._fvc_cache[key] = load_fvc_by_index(self.paths.fvc_dir, date, "FVC")
        return self._fvc_cache[key]

    def load_lst(self, date: datetime) -> np.ndarray:
        key = date.strftime("%Y%m%d")
        if key not in self._lst_cache:
            self._lst_cache[key] = load_lst_daily(self.paths.lst_dir, date, "LST")
        return self._lst_cache[key]

    def load_all_for_date(self, date: datetime) -> AuxData:
        """
        Load all auxiliary variables for a given date.
        """
        hveg = self.load_hveg()
        agb = self.load_agb()
        igbp = self.load_igbp(date.year)
        lai = self.load_lai(date)
        fvc = self.load_fvc(date)
        lst_k = self.load_lst(date)
        return AuxData(hveg=hveg, agb=agb, igbp=igbp, lai=lai, fvc=fvc, lst_k=lst_k)
