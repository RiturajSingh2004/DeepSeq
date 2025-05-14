import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Protein Analysis Platform"
    
    # Model paths
    MODELS_DIR: Path = BASE_DIR / "app" / "models" / "pretrained"
    EMBEDDING_MODEL_PATH: Path = MODELS_DIR / "embedding_model"
    PROPERTY_MODEL_PATH: Path = MODELS_DIR / "property_model"
    CLASSIFICATION_MODEL_PATH: Path = MODELS_DIR / "classification_model"
    STABILITY_MODEL_PATH: Path = MODELS_DIR / "stability_model"
    
    # AlphaFold settings
    ALPHAFOLD_ENABLED: bool = True
    ALPHAFOLD_DATA_DIR: Path = Path("/opt/alphafold_data")
    ALPHAFOLD_PARAMS_DIR: Path = ALPHAFOLD_DATA_DIR / "params"
    
    # Compute settings
    USE_GPU: bool = True
    NUM_THREADS: int = 4
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list[str] = ["fasta", "txt", "csv"]
    
    # Data storage
    TEMP_DIR: Path = BASE_DIR / "temp"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # Cache settings
    CACHE_ENABLED: bool = True
    CACHE_DIR: Path = BASE_DIR / "cache"
    CACHE_EXPIRY: int = 60 * 60 * 24 * 7  # 7 days in seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.MODELS_DIR, settings.TEMP_DIR, settings.RESULTS_DIR, settings.CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)