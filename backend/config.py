"""TransitVision AI - Backend Configuration & Settings."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class BackendSettings(BaseSettings):
    """Application backend configuration parameters."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "TransitVision AI"
    app_version: str = "1.0.0"
    app_description: str = "Production-style Autonomous MLOps & Real-Time ETA Intelligence Platform"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Simulation defaults
    default_replay_speed: float = 1.0
    simulation_tick_interval_sec: float = 0.5
    max_recent_events_buffer: int = 500
    
    # API prefix
    api_v1_prefix: str = "/api"


settings = BackendSettings()

