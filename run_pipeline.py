"""
Script: run_pipeline.py
Propósito: Script principal de orquestación de datos de la fase 3 (Modelado Final).

Pipeline:
1. Carga dataset de operaciones limpio y base de noticias con NLP Inference.
2. Ejecuta transformaciones con FeatureEngineer.
3. Alimenta y ajusta un modelo Cox Proportional Hazards con censura.
4. Exporta las ponderaciones y el logaritmo de verosimilitud (P-Values).
5. Genera el Prototipo Visual del Global Supply Chain Resilience Index. (MVP)

Uso: 
    python run_pipeline.py
"""

import logging
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.feature_engineering import FeatureEngineer
from src.models.survival_model import SupplyChainSurvivalModel
from src.visualization.map_builder import SupplyChainMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MLOpsOrchestrator")

def main():
    logger.info("=" * 65)
    logger.info(" 🚀 INICIANDO PIPELINE MLOPS Y VISUALIZACIÓN (FASE 3 & FINAL)")
    logger.info("=" * 65)

    base_dir = Path(__file__).parent
    processed_dir = base_dir / "data" / "processed"
    reports_dir = base_dir / "reports"
    
    # ── 1. Carga de Assets ─────────────────────────────────────────
    try:
        df_shipping_clean = pd.read_csv(processed_dir / "shipping_log_clean.csv")
        df_news_clean = pd.read_csv(processed_dir / "news_feed_enriched.csv")
    except FileNotFoundError:
        logger.critical("¡Peligro! Archivos no encontrados. Debiste correr el NLP primero.")
        sys.exit(1)

    # ── 2. Feature Engineering ─────────────────────────────────────
    fe = FeatureEngineer()
    df_master = fe.merge_datasets(df_shipping_clean, df_news_clean)
    df_master.to_csv(processed_dir / "master_table.csv", index=False)
    
    # ── 3. Entrenamiento estadístico (Cox PH) ──────────────────────
    survival_model = SupplyChainSurvivalModel()
    survival_model.train(df_master)
    print("\n")
    survival_model.print_summary()
    print("\n")
    
    # ── 4. Geoprocesamiento y Rasterización interactiva ────────────
    logger.info("\n4/4 - Compilando Global Map Dashboard (Folium Overlay)")
    map_engine = SupplyChainMap(master_table=df_master, news_enriched=df_news_clean)
    map_engine.build_map()
    
    output_path = reports_dir / "global_map.html"
    map_engine.save(output_path)
    
    print("=" * 70)
    print(f"✅ GSCRI MVP CONSTRUIDO.")
    print(f" -> RUTAS LOGISTICAS RENDERIZADAS: {map_engine.routes_drawn}")
    print(f" -> ALERTAS DE INTELIGENCIA DE RIESGO GEOSYNC'D: {map_engine.alerts_placed}")
    print(f" -> REPORTE HTML FINAL: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
