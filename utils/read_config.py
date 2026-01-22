import yaml
from typing import Dict, Any


def read_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}
