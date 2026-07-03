from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from stellar_evolution import SimulationConfig, simulate_structure


if __name__ == "__main__":
    result = simulate_structure(SimulationConfig(radial_steps=400))
    print(result["summary"])
