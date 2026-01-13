from typing import List, Optional
from pydantic import Field

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.types import Action, Observation, State

# Grid cell encoding:
# 0 = empty/ash, 1 = fuel (healthy), 2 = burning, 3 = firebreak, 4 = watered (damp)
# (You can tweak encodings, but keep them ints for compact obs.)

class WildfireAction(Action):
    """Action for the Wildfire environment."""
    # action: "break" (build firebreak), "water" (drop water), "wait"
    action: str = Field(..., description="Action type: break, water, or wait")
    x: Optional[int] = Field(default=None, description="X coordinate for action")
    y: Optional[int] = Field(default=None, description="Y coordinate for action")

class WildfireObservation(Observation):
    """Observation from the Wildfire environment."""
    grid: List[int] = Field(..., description="Flattened grid H*W, ints in {0..4}")
    width: int = Field(..., description="Grid width")
    height: int = Field(..., description="Grid height")
    step: int = Field(..., description="Current step number")
    wind_dir: str = Field(..., description="Wind direction: N, NE, E, SE, S, SW, W, NW, or CALM")
    humidity: float = Field(..., ge=0.0, le=1.0, description="Humidity level [0,1]")
    burning_count: int = Field(..., ge=0, description="Number of burning cells")
    burned_count: int = Field(..., ge=0, description="Total ash (0) cells (cumulative)")
    reward_hint: float = Field(default=0.0, description="Optional shaping info")
    remaining_water: int = Field(default=0, ge=0, description="Remaining water resources")
    remaining_breaks: int = Field(default=0, ge=0, description="Remaining firebreak resources")

class WildfireState(State):
    """Internal state for the Wildfire environment."""
    total_burned: int = Field(default=0, ge=0, description="Total cells burned")
    total_extinguished: int = Field(default=0, ge=0, description="Total fires extinguished")
    last_action: str = Field(default="reset", description="Last action taken")
    # For visibility / debugging (not required by core):
    width: int = Field(default=0, ge=0, description="Grid width")
    height: int = Field(default=0, ge=0, description="Grid height")
    wind_dir: str = Field(default="CALM", description="Wind direction")
    humidity: float = Field(default=0.25, ge=0.0, le=1.0, description="Humidity level")
    remaining_water: int = Field(default=20, ge=0, description="Remaining water resources")
    remaining_breaks: int = Field(default=50, ge=0, description="Remaining firebreak resources")
    # internal full grid as flattened ints
    grid: List[int] = Field(default_factory=list, description="Internal grid state")
    # burn timers for each cell (track how long cells have been burning/damp)
    burn_timers: List[int] = Field(default_factory=list, description="Burn timers for each cell")
