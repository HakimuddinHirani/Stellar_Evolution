from __future__ import annotations

from typing import Any

import numpy as np

from . import constants as const
from .color import blackbody_to_srgb, perceived_label
from .models import SimulationConfig
from .solver import simulate_structure


def _phase_for_age_fraction(age_fraction: float, mass_solar: float) -> str:
    if age_fraction < 0.12:
        return "Pre-main sequence"
    if age_fraction < 0.78:
        return "Main sequence"
    if mass_solar < 0.8:
        return "Late main sequence"
    if age_fraction < 0.91:
        return "Subgiant branch"
    return "Red giant branch"


def _display_phase(age_fraction: float, mass_solar: float, radius_solar: float, temperature: float) -> str:
    if age_fraction < 0.12:
        return "Pre-main sequence"
    if age_fraction < 0.78:
        return "Main sequence"
    if mass_solar >= 7.0 and temperature >= 9000.0 and radius_solar >= 2.0:
        return "Blue giant"
    if mass_solar >= 7.0 and radius_solar >= 10.0:
        return "Supergiant branch"
    if mass_solar < 0.8:
        return "Late main sequence"
    if age_fraction < 0.91:
        return "Subgiant branch"
    return "Red giant branch"


def _stellar_scaling(age_fraction: float, mass_solar: float) -> tuple[float, float, float]:
    """Return radius, luminosity, and effective temperature in solar units/K."""

    phase = _phase_for_age_fraction(age_fraction, mass_solar)
    base_luminosity = max(mass_solar, 0.08) ** 3.5
    base_radius = max(mass_solar, 0.08) ** 0.8

    if phase == "Pre-main sequence":
        t = age_fraction / 0.12
        radius = base_radius * (2.6 - 1.6 * t)
        luminosity = base_luminosity * (1.7 - 0.7 * t)
    elif phase == "Main sequence":
        t = (age_fraction - 0.12) / 0.66
        radius = base_radius * (1.0 + 0.35 * t)
        luminosity = base_luminosity * (1.0 + 0.75 * t)
    elif phase == "Late main sequence":
        t = (age_fraction - 0.78) / 0.22
        radius = base_radius * (1.35 + 0.35 * t)
        luminosity = base_luminosity * (1.75 + 0.55 * t)
    elif phase == "Subgiant branch":
        t = (age_fraction - 0.78) / 0.13
        radius = base_radius * (1.35 + 2.2 * t)
        luminosity = base_luminosity * (1.75 + 8.0 * t)
    else:
        t = (age_fraction - 0.91) / 0.09
        radius = base_radius * (3.55 + 32.0 * t)
        luminosity = base_luminosity * (9.75 + 240.0 * t)

    temperature = 5772.0 * (luminosity / max(radius**2, 1.0e-9)) ** 0.25
    return radius, luminosity, temperature


def generate_evolution_timeline(config: SimulationConfig, frames: int = 180) -> dict[str, Any]:
    """Generate fast frontend frames around the structure solver output.

    The full stellar-structure solve anchors the model. The timeline is then
    built from explainable stellar scaling relations so slider movement is
    instant in the browser.
    """

    config.validate()
    frames = min(max(int(frames), 24), 360)
    structure = simulate_structure(config)

    mass_solar = config.mass_solar
    main_sequence_years = 1.0e10 * mass_solar / max(mass_solar**3.5, 1.0e-6)
    total_years = 1.18 * main_sequence_years

    timeline = []
    for index, age_fraction in enumerate(np.linspace(0.0, 1.0, frames)):
        radius_solar, luminosity_solar, temperature = _stellar_scaling(age_fraction, mass_solar)
        color = blackbody_to_srgb(temperature)
        timeline.append(
            {
                "index": index,
                "age_fraction": float(age_fraction),
                "age_years": float(age_fraction * total_years),
                "phase": _display_phase(
                    float(age_fraction),
                    mass_solar,
                    radius_solar,
                    temperature,
                ),
                "radius_solar": float(radius_solar),
                "luminosity_solar": float(luminosity_solar),
                "effective_temperature": float(temperature),
                "color": {
                    "hex": color.hex,
                    "srgb": color.srgb,
                    "xyz": color.xyz,
                    "label": perceived_label(temperature),
                },
            }
        )

    return {
        "structure_summary": structure["summary"],
        "timeline": timeline,
    }
