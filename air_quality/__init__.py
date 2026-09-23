"""air-quality-modeling — Air pollution dispersion and source inversion"""

from .health import risk_probability
from .monitor import persistence
from .warning import aggregate, forecast, forecast_interval

__version__ = "0.1.0"

__all__ = [
    "aggregate",
    "forecast",
    "forecast_interval",
    "persistence",
    "risk_probability",
]
