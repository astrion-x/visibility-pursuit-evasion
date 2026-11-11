"""Physical parameters, numerical settings and scene stream identifiers."""

from dataclasses import dataclass
import json
from pathlib import Path

SCENE_CODES = {"A": 1, "B": 10, "C": 18, "D": 22}
METHODS = (
    "full",
    "proportional_navigation",
    "pure",
    "constant_bearing",
    "potential_field",
    "apollonius",
)
FIRST_SEED = 261494
PARAMETERS = dict(
    pursuer_speed="0.30",
    evader_speed="0.20",
    capture_radius="0.5",
    sample_period="0.1",
    observation_seconds=120,
    payoff_scale_m=50,
)


@dataclass(frozen=True)
class Settings:
    prediction_steps: int = 20
    directions: int = 16
    radii: tuple = (0.5, 1.0)
    map_spacing: float = 0.025
    sight_samples: int = 49
    equilibrium_gap_tolerance: float = 1e-7


def load_scenes():
    return json.loads(
        Path(__file__).with_name("scenes.json").read_text(encoding="utf-8")
    )
