"""Stable-Baselines3 callbacks used during training."""

from stable_baselines3.common.callbacks import BaseCallback


class SubrewardCallback(BaseCallback):
    """Write environment reward components to the SB3 logger."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            for name, value in info.get("subreward", {}).items():
                self.logger.record(f"subreward/{name}", value)
        return True
