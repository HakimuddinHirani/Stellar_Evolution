"""Backend package for the Stellar Evolution simulation."""

from .color import blackbody_to_srgb, perceived_label
from .models import Composition, SimulationConfig, SRGBColor
from .solver import simulate_structure

__all__ = [
    "Composition",
    "SimulationConfig",
    "SRGBColor",
    "blackbody_to_srgb",
    "perceived_label",
    "simulate_structure",
]
