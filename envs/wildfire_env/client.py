# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from .models import WildfireAction, WildfireObservation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from wildfire_env.models import WildfireAction, WildfireObservation

class WildfireEnv(EnvClient[WildfireAction, WildfireObservation, State]):
    def _step_payload(self, action: WildfireAction) -> dict:
        return {"action": action.action, "x": action.x, "y": action.y}

    def _parse_result(self, payload: dict) -> StepResult[WildfireObservation]:
        obs = WildfireObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )


def render_grid(obs: WildfireObservation) -> str:
    legend = {0:"⬛", 1:"🟩", 2:"🟥", 3:"🟫", 4:"🟦"}
    w, h = obs.width, obs.height
    g = obs.grid
    rows = []
    for y in range(h):
        rows.append("".join(legend.get(g[y*w+x], "?") for x in range(w)))
    meta = f"step={obs.step} wind={obs.wind_dir} hum={obs.humidity:.2f} burning={obs.burning_count} burned={obs.burned_count}"
    return "\n".join(rows + [meta])
