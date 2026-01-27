from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AuxPaths:
    canopy_height_h5: str = r"G:\data\CanopyHeight\CH.h5"
    igbp_dir: str = r"G:\data\MCD12C1 CMG\01Degree\h5"
    agb_dir: str = r"G:\data\ESACCI AGB\h5"
    lai_dir: str = r"G:\data\GLASS LAI\01nc"
    fvc_dir: str = r"G:\data\GLASS FVC\01nc"
    lst_dir: str = r"G:\data\MOD11C1 CMG LST\01h5"
