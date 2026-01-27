from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class PhysicsParams:
    # per band a,b
    a_C: float
    b_C: float
    a_X: float
    b_X: float
    a_Ku: float
    b_Ku: float

    # global canopy fraction
    delta: float

    # vegetation dielectric params (only fitting k_re,k_im here for minimal)
    k_re: float
    k_im: float


def params_to_vector(p: PhysicsParams, names: List[str]) -> np.ndarray:
    return np.array([getattr(p, n) for n in names], dtype=float)


def vector_to_params(x: np.ndarray, base: PhysicsParams, names: List[str]) -> PhysicsParams:
    d = base.__dict__.copy()
    for i, n in enumerate(names):
        d[n] = float(x[i])
    return PhysicsParams(**d)


def build_bounds(names: List[str], bounds_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    bounds_cfg example:
      {"a":[-2,2], "b":[0,5], "delta":[0,0.6], "k_re":[0,80], "k_im":[0,80]}
    """
    lo, hi = [], []
    for n in names:
        if n.startswith("a_"):
            b = bounds_cfg["a"]
        elif n.startswith("b_"):
            b = bounds_cfg["b"]
        elif n == "delta":
            b = bounds_cfg["delta"]
        elif n == "k_re":
            b = bounds_cfg["k_re"]
        elif n == "k_im":
            b = bounds_cfg["k_im"]
        else:
            raise ValueError(f"Unknown param name: {n}")
        lo.append(float(b[0])); hi.append(float(b[1]))
    return np.array(lo, dtype=float), np.array(hi, dtype=float)
