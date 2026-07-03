from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import jax.numpy as jnp
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "JAX is required for the Stellar Evolution backend. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

from . import constants as const
from .models import SimulationConfig
from .physics import (
    density_from_pressure_temperature,
    mean_molecular_weight,
    opacity_kramers_plus_scattering,
    pp_chain_energy_rate,
    rk4_step,
)


def _initial_state(config: SimulationConfig, radius_0: float, mu: float) -> jnp.ndarray:
    rho_c = float(
        density_from_pressure_temperature(
            config.central_pressure,
            config.central_temperature,
            mu,
        )
    )
    mass_0 = 4.0 * math.pi * radius_0**3 * rho_c / 3.0
    epsilon_0 = float(pp_chain_energy_rate(rho_c, config.central_temperature, config.composition.hydrogen))
    luminosity_0 = mass_0 * epsilon_0
    return jnp.array([mass_0, config.central_pressure, luminosity_0, config.central_temperature])


def simulate_structure(config: SimulationConfig) -> dict[str, Any]:
    """Solve one static radial stellar model.

    This integrates outward from a small central radius. The input central
    pressure and temperature act as shooting parameters; later project phases
    can add automatic shooting to satisfy surface boundary conditions.
    """

    config.validate()

    radius_star = config.radius_solar * const.R_SUN
    target_mass = config.mass_solar * const.M_SUN
    radius_0 = radius_star / config.radial_steps
    dr = (radius_star - radius_0) / (config.radial_steps - 1)

    composition = config.composition
    mu = mean_molecular_weight(composition.hydrogen, composition.helium, composition.metals)
    state = _initial_state(config, radius_0, mu)

    radii = []
    states = []
    stop_reason = "completed_requested_radius"
    pressure_floor = config.central_pressure * config.surface_pressure_fraction

    for step in range(config.radial_steps):
        radius = radius_0 + step * dr
        mass, pressure, _, temperature = [float(value) for value in state]

        if pressure <= pressure_floor:
            stop_reason = "surface_pressure_floor_reached"
            break
        if temperature <= 1.0:
            stop_reason = "temperature_floor_reached"
            break
        if mass >= target_mass:
            stop_reason = "target_mass_reached"
            break

        radii.append(radius)
        states.append(np.asarray(state, dtype=float))
        state = rk4_step(radius, state, dr, mu, composition.hydrogen, composition.metals)
        state = jnp.maximum(state, jnp.array([0.0, 1.0, 0.0, 1.0]))

    if not states:
        raise RuntimeError("Simulation stopped before producing any radial profile points.")

    states_array = np.vstack(states)
    radii_array = np.asarray(radii)
    mass = states_array[:, 0]
    pressure = states_array[:, 1]
    luminosity = states_array[:, 2]
    temperature = states_array[:, 3]

    density = np.asarray(
        density_from_pressure_temperature(jnp.asarray(pressure), jnp.asarray(temperature), mu),
        dtype=float,
    )
    opacity = np.asarray(
        opacity_kramers_plus_scattering(
            jnp.asarray(density),
            jnp.asarray(temperature),
            composition.hydrogen,
            composition.metals,
        ),
        dtype=float,
    )
    epsilon = np.asarray(
        pp_chain_energy_rate(jnp.asarray(density), jnp.asarray(temperature), composition.hydrogen),
        dtype=float,
    )

    final_index = -1
    summary = {
        "stop_reason": stop_reason,
        "points": int(len(radii_array)),
        "mean_molecular_weight": mu,
        "final_radius_solar": float(radii_array[final_index] / const.R_SUN),
        "final_mass_solar": float(mass[final_index] / const.M_SUN),
        "final_luminosity_solar": float(luminosity[final_index] / const.L_SUN),
        "final_pressure": float(pressure[final_index]),
        "final_temperature": float(temperature[final_index]),
        "central_density": float(density[0]),
    }

    return {
        "summary": summary,
        "profiles": {
            "radius_cm": radii_array.tolist(),
            "radius_solar": (radii_array / const.R_SUN).tolist(),
            "mass_g": mass.tolist(),
            "mass_solar": (mass / const.M_SUN).tolist(),
            "pressure": pressure.tolist(),
            "temperature": temperature.tolist(),
            "luminosity": luminosity.tolist(),
            "luminosity_solar": (luminosity / const.L_SUN).tolist(),
            "density": density.tolist(),
            "opacity": opacity.tolist(),
            "epsilon": epsilon.tolist(),
        },
    }
