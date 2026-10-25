"""Method settings, independent of data paths and labels."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class FGGMConfig:
    h: float = 20.0
    alpha: float = 0.01
    confidence_threshold: float = 0.1
    logit_scale: float = 100.0
    geometry_weight: float = 0.5
    gamma: float = 2.0
    temperature: float = 2.25
    frequency_rate: float | None = None  # None means 1 / (10 * classes)
    min_history: int = 128
    update_interval: int = 128
    covariance_floor: float = 1e-8
    denominator_floor: float = 1e-10
    normalization: str = "guarded"  # "legacy" preserves the archived division

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if self.h <= 0 or self.temperature <= 0 or self.logit_scale <= 0:
            raise ValueError("h, temperature and logit_scale must be positive")
        if not 0 <= self.alpha <= 1 or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("alpha and confidence_threshold must be in [0, 1]")
        if not 0 <= self.geometry_weight <= 1 or self.gamma < 0:
            raise ValueError("geometry_weight must be in [0, 1] and gamma >= 0")
        if self.frequency_rate is not None and not 0 < self.frequency_rate < 1:
            raise ValueError("frequency_rate must be in (0, 1) to retain positive historical mass")
        if self.min_history < 1 or self.update_interval < 1:
            raise ValueError("History and update interval must be positive integers")
        if not isinstance(self.min_history, int) or not isinstance(self.update_interval, int):
            raise ValueError("History and update interval must be integers")
        if self.covariance_floor <= 0 or self.denominator_floor <= 0:
            raise ValueError("Numerical floors must be positive")
        if self.normalization not in {"guarded", "legacy"}:
            raise ValueError("normalization must be guarded or legacy")

    @classmethod
    def from_json(cls, path):
        content = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        known = {f.name for f in fields(cls)}
        if not isinstance(content, dict) or set(content) - known:
            raise ValueError("Config must be an object containing only FGGMConfig fields")
        return cls(**content)

    def to_dict(self):
        return asdict(self)
