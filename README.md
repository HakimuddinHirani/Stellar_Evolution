# Stellar Evolution

Backend-first simulation project for the Ad Astra Research and Tech team.

The backend implements the four stellar-structure equations highlighted in the
research paper:

1. Mass continuity: `dM/dr = 4*pi*r^2*rho`
2. Hydrostatic equilibrium: `dP/dr = -G*M*rho/r^2`
3. Energy generation: `dL/dr = 4*pi*r^2*rho*epsilon`
4. Radiative transport: `dT/dr = -(3*kappa*rho*L)/(16*pi*a*c*T^3*r^2)`

Density is closed with an ideal-gas plus radiation-pressure equation of state.
The project uses JAX for the numerical physics kernels and Flask for the API
that the frontend can call later.

## Project Layout

```text
Project_Code/
  backend/
    app.py                         Flask entry point
    stellar_evolution/
      constants.py                 Physical constants
      models.py                    Input/output data models
      physics.py                   JAX equation kernels
      solver.py                    RK4 stellar-structure solver
      color.py                     Blackbody -> CIE XYZ -> sRGB pipeline
      evolution.py                 Precomputed evolution timeline frames
      api.py                       Flask routes
  frontend/
    index.html                     User interface
    styles.css                     Layout and visual design
    app.js                         Slider, API calls, and chart rendering
  examples/
    run_simulation.py              CLI smoke example
  requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows, the CPU build of JAX is enough for this project.

## Run The Backend

```powershell
python backend\app.py
```

Then test:

```powershell
curl http://127.0.0.1:5000/health
```

Open the frontend in the browser:

```text
http://127.0.0.1:5000
```

## Example Simulation

```powershell
python examples\run_simulation.py
```

## API

`POST /simulate`

```json
{
  "mass_solar": 1.0,
  "radius_solar": 1.0,
  "central_temperature": 1.55e7,
  "central_pressure": 2.45e17,
  "composition": {
    "hydrogen": 0.70,
    "helium": 0.28,
    "metals": 0.02
  },
  "radial_steps": 1200
}
```

The response contains radial profiles for radius, enclosed mass, pressure,
temperature, luminosity, density, opacity, and energy generation rate.

`POST /evolution`

Uses the same input fields as `/simulate`, plus an optional `frames` field.
It returns precomputed timeline frames for the frontend slider:

```json
{
  "mass_solar": 1.0,
  "radius_solar": 1.0,
  "central_temperature": 1.55e7,
  "central_pressure": 2.45e17,
  "composition": {
    "hydrogen": 0.70,
    "helium": 0.28,
    "metals": 0.02
  },
  "radial_steps": 800,
  "frames": 180
}
```

`POST /color`

```json
{
  "temperature": 5772
}
```

Returns the perceived blackbody color as CIE XYZ, linear RGB, sRGB, and hex.

## Color Pipeline

The visual color path follows the astronomy-to-display idea:

1. Treat the star as a blackbody radiator with an effective temperature.
2. Estimate display sRGB from color temperature for stable browser rendering.
3. Convert linear sRGB into CIE XYZ values for the perception readout.
4. Apply the sRGB gamma curve for final screen color.

This is why the frontend can show how a cool red giant, Sun-like star, or hot
blue-white star would be perceived by human vision and displayed on a screen.

## Scientific Scope

This is a teaching-grade first backend. It solves a simplified 1D, spherical,
radiative stellar model with approximate microphysics:

- ideal-gas plus radiation-pressure EOS
- Kramers opacity plus electron scattering
- simplified proton-proton-chain energy production
- fixed composition during one structure solve
- approximate CIE matching functions for compact educational color conversion
- lightweight stellar scaling relations for scrub-friendly timeline frames

The code is organized so the team can later replace these approximations with
tabulated EOS/opacity data, convection, more detailed nuclear networks, and
time evolution.
