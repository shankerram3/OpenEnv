# server/app.py
import os

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app
    from ..models import WildfireAction, WildfireObservation
    from .wildfire_environment import WildfireEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from wildfire_env.models import WildfireAction, WildfireObservation
    from wildfire_env.server.wildfire_environment import WildfireEnvironment

# Create the app with web interface and README integration
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(WildfireEnvironment, WildfireAction, WildfireObservation, env_name="wildfire_env")


def main():
    """Main entry point for running the server."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
