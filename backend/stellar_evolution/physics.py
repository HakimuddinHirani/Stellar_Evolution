"""JAX-powered physics kernels for the stellar-structure equations."""

try:
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    from jax import Array
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "JAX is required for the Stellar Evolution backend. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

from . import constants as const


def mean_molecular_weight(hydrogen: float, helium: float, metals: float) -> float:
    """Fully ionized gas approximation: 1/mu = 2X + 3Y/4 + Z/2."""

    inverse_mu = 2.0 * hydrogen + 0.75 * helium + 0.5 * metals
    return 1.0 / inverse_mu


@jax.jit
def density_from_pressure_temperature(
    pressure: float,
    temperature: float,
    mu: float,
) -> Array:
    """EOS closure: gas pressure plus radiation pressure."""

    radiation_pressure = const.A_RAD * temperature**4 / 3.0
    gas_pressure = jnp.maximum(pressure - radiation_pressure, 1.0e-30)
    return gas_pressure * mu * const.M_H / (const.K_B * temperature)


@jax.jit
def opacity_kramers_plus_scattering(
    density: float,
    temperature: float,
    hydrogen: float,
    metals: float,
) -> Array:
    """Approximate Rosseland opacity in cm^2/g."""

    electron_scattering = 0.2 * (1.0 + hydrogen)
    kramers = 4.0e25 * (1.0 + hydrogen) * (metals + 1.0e-4) * density * temperature**-3.5
    return electron_scattering + kramers


@jax.jit
def pp_chain_energy_rate(
    density: float,
    temperature: float,
    hydrogen: float,
) -> Array:
    """Simplified proton-proton chain energy generation in erg g^-1 s^-1."""

    t6 = temperature / 1.0e6
    exponent = -33.8 * jnp.power(jnp.maximum(t6, 1.0e-6), -1.0 / 3.0)
    return 2.4e4 * density * hydrogen**2 * t6 ** (-2.0 / 3.0) * jnp.exp(exponent)


@jax.jit
def stellar_structure_derivatives(
    radius: float,
    state: jnp.ndarray,
    mu: float,
    hydrogen: float,
    metals: float,
) -> jnp.ndarray:
    """Return d[M, P, L, T] / dr for the four stellar-structure equations."""

    mass, pressure, luminosity, temperature = state
    safe_radius = jnp.maximum(radius, 1.0)
    safe_temperature = jnp.maximum(temperature, 1.0)
    safe_pressure = jnp.maximum(pressure, 1.0)

    density = density_from_pressure_temperature(safe_pressure, safe_temperature, mu)
    opacity = opacity_kramers_plus_scattering(density, safe_temperature, hydrogen, metals)
    epsilon = pp_chain_energy_rate(density, safe_temperature, hydrogen)

    d_mass = 4.0 * const.PI * safe_radius**2 * density
    d_pressure = -const.G * mass * density / safe_radius**2
    d_luminosity = 4.0 * const.PI * safe_radius**2 * density * epsilon
    d_temperature = (
        -3.0
        * opacity
        * density
        * luminosity
        / (16.0 * const.PI * const.A_RAD * const.C * safe_temperature**3 * safe_radius**2)
    )

    return jnp.array([d_mass, d_pressure, d_luminosity, d_temperature])


@jax.jit
def rk4_step(
    radius: float,
    state: jnp.ndarray,
    dr: float,
    mu: float,
    hydrogen: float,
    metals: float,
) -> jnp.ndarray:
    """One fourth-order Runge-Kutta radial integration step."""

    k1 = stellar_structure_derivatives(radius, state, mu, hydrogen, metals)
    k2 = stellar_structure_derivatives(radius + 0.5 * dr, state + 0.5 * dr * k1, mu, hydrogen, metals)
    k3 = stellar_structure_derivatives(radius + 0.5 * dr, state + 0.5 * dr * k2, mu, hydrogen, metals)
    k4 = stellar_structure_derivatives(radius + dr, state + dr * k3, mu, hydrogen, metals)
    return state + (dr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
