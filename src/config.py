"""
Módulo: config.py
Responsabilidad: Configuración centralizada del proyecto usando variables de entorno.

Todas las constantes y parámetros globales del proyecto se definen aquí.
Se usa python-dotenv para cargar desde un archivo .env en desarrollo.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Cargar variables de entorno desde .env si existe
load_dotenv()

# ─── Rutas del Proyecto ────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
LOGS_DIR: Path = ROOT_DIR / "logs"
MODELS_DIR: Path = ROOT_DIR / "src" / "models" / "artifacts"

# Crear directorios si no existen
for _dir in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, LOGS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"),
    ],
)


# ─── Configuración de APIs ──────────────────────────────────────────────────────
@dataclass
class APIConfig:
    """Credenciales y endpoints de APIs externas."""
    world_bank_base_url: str = "https://api.worldbank.org/v2"
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2"
    comtrade_base_url: str = "https://comtradeapi.un.org/data/v1"
    comtrade_api_key: str = field(default_factory=lambda: os.getenv("COMTRADE_API_KEY", ""))
    request_timeout: int = 30
    max_retries: int = 3


# ─── Configuración del Modelo ───────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Hiperparámetros y configuración del modelo de supervivencia."""
    cox_penalizer: float = float(os.getenv("COX_PENALIZER", "0.1"))
    duration_col: str = "duration_days"
    event_col: str = "disruption_occurred"
    test_size: float = 0.2
    random_state: int = 42


# ─── Configuración del Índice ───────────────────────────────────────────────────
@dataclass
class IndexConfig:
    """Parámetros de cálculo del GSCRI."""
    geopolitical_weight: float = 0.30
    climate_weight: float = 0.25
    supplier_concentration_weight: float = 0.20
    logistics_weight: float = 0.15
    sentiment_weight: float = 0.10


# Instancias globales (singleton-like)
API_CONFIG = APIConfig()
MODEL_CONFIG = ModelConfig()
INDEX_CONFIG = IndexConfig()
