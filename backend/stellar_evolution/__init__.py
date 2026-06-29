"""Backend package for the Stellar Evolution simulation."""

from .models import Composition, SimulationConfig
from .solver import simulate_structure

__all__ = ["Composition", "SimulationConfig", "simulate_structure"]
