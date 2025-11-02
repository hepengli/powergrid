"""Configuration loader utility for loading YAML environment configs."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_name: str) -> Dict[str, Any]:
    """Load environment configuration from YAML file.

    Args:
        config_name: Name of the config file (with or without .yml extension)

    Returns:
        Dictionary containing the environment configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    # Get config directory (where this file is located)
    config_dir = Path(__file__).parent

    # Add .yml extension if not present
    if not config_name.endswith('.yml'):
        config_name = f"{config_name}.yml"

    config_path = config_dir / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_available_configs() -> list[str]:
    """Get list of available configuration files.

    Returns:
        List of config file names (without .yml extension)
    """
    config_dir = Path(__file__).parent
    return [f.stem for f in config_dir.glob('*.yml')]


if __name__ == "__main__":
    # Test the loader
    print("Available configs:", get_available_configs())

    config = load_config('ieee34_ieee13')
    print("\nLoaded config keys:", list(config.keys()))
    print(f"Number of microgrids: {len(config.get('mg_configs', []))}")
    print(f"Dataset path: {config.get('dataset_path')}")
    print(f"Training mode: {config.get('train')}")
