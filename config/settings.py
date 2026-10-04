"""Configuration loader and environment manager for TransitVision AI."""
import os
from pathlib import Path
from typing import Any, Dict
import yaml

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Path to the primary YAML configuration file
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config(config_file: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable fallback."""
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths to absolute paths
    if "PATHS" in config:
        for key, rel_path in config["PATHS"].items():
            config["PATHS"][key] = str((ROOT_DIR / rel_path).resolve())
            
    return config


# Global configuration instance
CONFIG = load_config()

# Convenience path shortcuts
RAW_DATA_DIR = Path(CONFIG["PATHS"]["RAW_DATA_DIR"])
INTERIM_DATA_DIR = Path(CONFIG["PATHS"]["INTERIM_DATA_DIR"])
PROCESSED_DATA_DIR = Path(CONFIG["PATHS"]["PROCESSED_DATA_DIR"])
METADATA_DIR = Path(CONFIG["PATHS"]["METADATA_DIR"])
MODELS_DIR = Path(CONFIG["PATHS"]["MODELS_DIR"])
REGISTRY_DIR = Path(CONFIG["PATHS"]["REGISTRY_DIR"])
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
SIMULATION_DATA_DIR = ROOT_DIR / "data" / "simulation"
