"""Data-models for the Stellar Evolution simulation.

All public models are plain Python dataclasses so they remain importable
without installing Pydantic.  The ``validate`` method on ``SimulationConfig``
performs the domain-level checks that the API layer re-surfaces as 400
responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Colour output model
# ---------------------------------------------------------------------------


@dataclass
class SRGBColor:
    """Perceived blackbody colour in several representations.

    Attributes
    ----------
    hex:
        HTML hex string, e.g. ``"#fff5e0"``.
    srgb:
        Gamma-encoded sRGB triple, each value in [0.0, 1.0].
    linear_srgb:
        Linear (pre-gamma) sRGB triple, each value in [0.0, 1.0].
    xyz:
        CIE 1931 XYZ tristimulus values normalised to Y = 1.
    label:
        Human-readable perceived colour name, e.g. ``"yellow-white"``.
    """

    hex: str
    srgb: List[float]
    linear_srgb: List[float]
    xyz: List[float]
    label: str


# ---------------------------------------------------------------------------
# Simulation input models
# ---------------------------------------------------------------------------


@dataclass
class Composition:
    """Mass-fraction composition of the stellar gas.

    The three fractions must sum to 1 within a tolerance of 0.01.
    """

    hydrogen: float = 0.70
    helium: float = 0.28
    metals: float = 0.02

    def validate(self) -> None:
        total = self.hydrogen + self.helium + self.metals
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Composition fractions must sum to 1.0; got {total:.4f}. "
                "Adjust hydrogen, helium, or metals."
            )
        for name, value in [
            ("hydrogen", self.hydrogen),
            ("helium", self.helium),
            ("metals", self.metals),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Composition fraction '{name}' must be in [0, 1]; got {value}."
                )

    @classmethod
    def from_dict(cls, data: dict) -> "Composition":
        return cls(
            hydrogen=float(data.get("hydrogen", 0.70)),
            helium=float(data.get("helium", 0.28)),
            metals=float(data.get("metals", 0.02)),
        )

    def to_dict(self) -> dict:
        return {
            "hydrogen": self.hydrogen,
            "helium": self.helium,
            "metals": self.metals,
        }


@dataclass
class SimulationConfig:
    """Full specification of a stellar-structure simulation run.

    Parameters
    ----------
    mass_solar:
        Total stellar mass in solar masses (0.1 – 25).
    radius_solar:
        Initial stellar radius in solar radii (0.1 – 100).
    central_temperature:
        Central temperature in Kelvin (1e6 – 8e8).
    central_pressure:
        Central pressure in dyne cm⁻² (1e12 – 1e22).
    composition:
        Mass-fraction breakdown of hydrogen, helium, and metals.
    radial_steps:
        Number of radial integration steps (24 – 4000).
    frames:
        Number of timeline frames for the evolution slider (24 – 360).
    surface_pressure_fraction:
        Stop the radial integration when pressure drops below this
        fraction of the central pressure. Default 1e-6.
    """

    mass_solar: float = 1.0
    radius_solar: float = 1.0
    central_temperature: float = 1.55e7
    central_pressure: float = 2.45e17
    composition: Composition = field(default_factory=Composition)
    radial_steps: int = 1200
    frames: int = 180
    surface_pressure_fraction: float = 1.0e-6

    def validate(self) -> None:
        if not (0.1 <= self.mass_solar <= 25.0):
            raise ValueError(
                f"mass_solar must be in [0.1, 25]; got {self.mass_solar}."
            )
        if not (0.1 <= self.radius_solar <= 100.0):
            raise ValueError(
                f"radius_solar must be in [0.1, 100]; got {self.radius_solar}."
            )
        if not (1.0e6 <= self.central_temperature <= 8.0e8):
            raise ValueError(
                f"central_temperature must be in [1e6, 8e8] K; "
                f"got {self.central_temperature:.3e}."
            )
        if not (1.0e12 <= self.central_pressure <= 1.0e22):
            raise ValueError(
                f"central_pressure must be in [1e12, 1e22] dyne/cm²; "
                f"got {self.central_pressure:.3e}."
            )
        if not (24 <= self.radial_steps <= 4000):
            raise ValueError(
                f"radial_steps must be in [24, 4000]; got {self.radial_steps}."
            )
        self.composition.validate()

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        composition_raw = data.get("composition", {})
        composition = (
            Composition.from_dict(composition_raw)
            if isinstance(composition_raw, dict)
            else Composition()
        )
        return cls(
            mass_solar=float(data.get("mass_solar", 1.0)),
            radius_solar=float(data.get("radius_solar", 1.0)),
            central_temperature=float(data.get("central_temperature", 1.55e7)),
            central_pressure=float(data.get("central_pressure", 2.45e17)),
            composition=composition,
            radial_steps=int(data.get("radial_steps", 1200)),
            frames=int(data.get("frames", 180)),
            surface_pressure_fraction=float(
                data.get("surface_pressure_fraction", 1.0e-6)
            ),
        )
