"""
Smart Adaptive Street Lighting - RL Environment
SDG 7 (Clean Energy) + SDG 11 (Sustainable Cities)
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BRIGHTNESS_LEVELS = {0: "Off", 1: "Dim", 2: "Medium", 3: "Full"}
BRIGHTNESS_WATTS  = {0: 0, 1: 30, 2: 60, 3: 100}   # watt equivalents
NUM_ACTIONS       = 4   # Off / Dim / Medium / Full
NUM_TIME_BINS     = 24  # hours 0-23
MAX_PEDESTRIANS   = 10  # max pedestrians in state

# State discretisation sizes
TIME_BINS    = 24
PED_BINS     = 5   # 0-2, 3-4, 5-6, 7-8, 9-10
BRIGHT_BINS  = 4   # mirrors actions
STATE_SIZE   = TIME_BINS * PED_BINS * BRIGHT_BINS


@dataclass
class Step:
    """Named return type for environment.step()"""
    state:      Tuple[int, int, int]
    reward:     float
    done:       bool
    info:       Dict[str, Any]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def discretise_pedestrians(count: int) -> int:
    """Map raw pedestrian count → bin index [0, PED_BINS)."""
    thresholds = [0, 2, 4, 6, 8, 10]
    for i, t in enumerate(thresholds[1:], start=0):
        if count <= t:
            return i
    return PED_BINS - 1


def state_to_index(state: Tuple[int, int, int]) -> int:
    """Flatten (time, ped_bin, brightness) → single integer index."""
    t, p, b = state
    return t * (PED_BINS * BRIGHT_BINS) + p * BRIGHT_BINS + b


# ---------------------------------------------------------------------------
# Pedestrian demand model
# ---------------------------------------------------------------------------
def pedestrian_demand(hour: int) -> int:
    """
    Simulate realistic pedestrian counts per hour.
      - Low at night (0-5)
      - Morning rush (7-9)
      - Lunchtime bump (12-13)
      - Evening rush (17-20)
    """
    base = [
        0, 0, 0, 0, 0, 0,   # 00-05
        1, 4, 7, 6, 3, 2,   # 06-11
        5, 5, 3, 2, 2, 6,   # 12-17
        8, 9, 7, 4, 2, 1,   # 18-23
    ]
    noise = random.randint(-1, 2)
    return max(0, min(MAX_PEDESTRIANS, base[hour] + noise))


# ---------------------------------------------------------------------------
# Core Environment
# ---------------------------------------------------------------------------
class StreetLightEnv:
    """
    Tabular RL environment for adaptive street-light control.

    Observation (state):
        (hour [0-23], pedestrian_bin [0-4], brightness_action [0-3])

    Actions:
        0 = Off  | 1 = Dim  | 2 = Medium  | 3 = Full

    Reward design:
        +10  → pedestrians present AND brightness is Medium or Full
        -10  → no pedestrians AND brightness is Full
        -5   → energy waste penalty proportional to wattage when no pedestrians
        -2   → dim when pedestrians present (safety concern)
        0    → otherwise neutral
    """

    def __init__(self,
                 episode_length: int = 24,
                 seed: int | None = None):
        self.episode_length = episode_length
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.action_space_n   = NUM_ACTIONS
        self.observation_size = STATE_SIZE

        self._hour           = 0
        self._pedestrians    = 0
        self._brightness     = 0
        self._step_count     = 0
        self._total_energy   = 0.0
        self._unnecessary_steps = 0

    # ------------------------------------------------------------------
    def reset(self, start_hour: int | None = None) -> Tuple[int, int, int]:
        """Reset episode. Returns initial state tuple."""
        self._hour       = start_hour if start_hour is not None else self.rng.randint(0, 23)
        self._pedestrians = pedestrian_demand(self._hour)
        self._brightness  = 0
        self._step_count  = 0
        self._total_energy = 0.0
        self._unnecessary_steps = 0
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Step:
        """Apply action, advance time, compute reward."""
        assert 0 <= action < NUM_ACTIONS, f"Invalid action {action}"

        self._brightness = action
        watts = BRIGHTNESS_WATTS[action]
        self._total_energy += watts

        reward = self._compute_reward(action, self._pedestrians, watts)

        # Track unnecessary full-brightness steps
        if self._pedestrians == 0 and action == 3:
            self._unnecessary_steps += 1

        # Advance time
        self._step_count += 1
        done = self._step_count >= self.episode_length

        if not done:
            self._hour = (self._hour + 1) % 24
            self._pedestrians = pedestrian_demand(self._hour)

        info = {
            "hour":         self._hour,
            "pedestrians":  self._pedestrians,
            "brightness":   BRIGHTNESS_LEVELS[action],
            "watts":        watts,
            "energy_total": self._total_energy,
            "unnecessary_steps": self._unnecessary_steps,
        }

        return Step(state=self._get_state(), reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    def _get_state(self) -> Tuple[int, int, int]:
        ped_bin = discretise_pedestrians(self._pedestrians)
        return (self._hour, ped_bin, self._brightness)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_reward(action: int, pedestrians: int, watts: float) -> float:
        reward = 0.0

        if pedestrians > 0:
            if action in (2, 3):          # Medium or Full → safe
                reward += 10.0
            elif action == 1:             # Dim → marginal safety
                reward -= 2.0
            else:                         # Off with pedestrians → dangerous
                reward -= 15.0
        else:
            if action == 3:               # Full brightness, nobody there
                reward -= 10.0
            # General energy waste penalty (proportional)
            reward -= (watts / 100.0) * 5.0

        return reward

    # ------------------------------------------------------------------
    @property
    def energy_used(self) -> float:
        return self._total_energy

    @property
    def unnecessary_lighting_pct(self) -> float:
        if self._step_count == 0:
            return 0.0
        return 100.0 * self._unnecessary_steps / self._step_count

    def render(self) -> str:
        """Human-readable current state."""
        ped_bin = discretise_pedestrians(self._pedestrians)
        return (
            f"Hour={self._hour:02d}:00  "
            f"Peds={self._pedestrians}  "
            f"Brightness={BRIGHTNESS_LEVELS[self._brightness]}  "
            f"Energy={self._total_energy:.0f}Wh"
        )
