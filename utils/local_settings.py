import os
import json

class ConfigManager:
    def __init__(self):
        self.config_dir = os.path.join(os.path.expanduser("~"), ".groq-python-agent")
        self.config_path = os.path.join(self.config_dir, "config.json")
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

    def _read_config(self) -> dict:
        if not os.path.exists(self.config_path):
            return {}
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _write_config(self, config: dict):
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def get_api_key(self) -> str | None:
        config = self._read_config()
        return config.get("api_key")

    def set_api_key(self, api_key: str):
        config = self._read_config()
        config["api_key"] = api_key
        self._write_config(config)

    def get_default_model(self) -> str | None:
        config = self._read_config()
        return config.get("default_model")

    def set_default_model(self, model: str):
        config = self._read_config()
        config["default_model"] = model
        self._write_config(config)