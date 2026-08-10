"""
AEGIS configuration loader.

Supports default configuration with optional YAML overrides.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .defaults import DEFAULT_CONFIG


class AEGISConfig:
    """Central configuration object for the AEGIS framework."""

    def __init__(self, values: Dict[str, Any] | None = None):
        self._values = deepcopy(DEFAULT_CONFIG)

        if values:
            self._deep_update(self._values, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AEGISConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML configuration files. "
                "Install it using: pip install pyyaml"
            ) from exc

        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}

        return cls(values)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a nested configuration value using dot notation.

        Example:
            config.get("framework.version")
        """

        current = self._values

        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default

            current = current[part]

        return current

    def as_dict(self) -> Dict[str, Any]:
        return deepcopy(self._values)

    @staticmethod
    def _deep_update(
        target: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> None:
        for key, value in updates.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                AEGISConfig._deep_update(target[key], value)
            else:
                target[key] = value