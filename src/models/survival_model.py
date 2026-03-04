"""
Módulo: survival_model.py
Responsabilidad: Entrenamiento, evaluación y predicción usando análisis de supervivencia.

Modelo principal: Cox Proportional Hazards (CoxPHFitter) de la librería lifelines.
Predictor para inferir los hazards ratios debidos al Voyage Risk Index NLP.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    from lifelines import CoxPHFitter
    _LIFELINES_AVAILABLE = True
except ImportError:
    logger.warning("lifelines no disponible. Instala con: pip install lifelines.")
    _LIFELINES_AVAILABLE = False


class SupplyChainSurvivalModel:
    """
    Modelo estadístico Cox PH configurado para predecir anomalías e interrupciones logísticas.
    """

    def __init__(
        self,
        duration_col: str = "delay_days",
        event_col: str = "event_observed",
        penalizer: float = 0.05,
    ) -> None:
        if not _LIFELINES_AVAILABLE:
            raise RuntimeError("lifelines no está instalado. Verifica requirements.txt.")

        self.duration_col = duration_col
        self.event_col = event_col
        
        # Penalización L2 (Ridge) para regularizar y lidiar con alta colinealidad
        self.model = CoxPHFitter(penalizer=penalizer)
        
        # Covariables críticas definidas
        self.covariates = [
            "voyage_risk_index", 
            "weather_severity", 
            "vessel_capacity_teu", 
            "geopolitical_risk"
        ]

    def train(self, df: pd.DataFrame) -> None:
        """
        Ajusta el modelo Cox a los datos de la Master Table.
        Se ignoran filas incompletas en covariables esenciales.
        """
        logger.info("Preparando tensores para Survival Analysis...")
        cols = self.covariates + [self.duration_col, self.event_col]
        
        # Filtro estricto de nulos solo sobre las features objetivo para no descartar censuradas (In Transit)
        train_df = df[cols].dropna()

        logger.info("Entrenando algoritmo Cox PH sobre %d registros validados.", len(train_df))
        
        try:
            self.model.fit(
                train_df,
                duration_col=self.duration_col,
                event_col=self.event_col,
                show_progress=False
            )
            logger.info("✔️ Entrenamiento finalizado. Concordance Index (Precisión MLOps): %.4f", self.model.concordance_index_)
        except Exception as exc:
            logger.error("Error catastrofico en ajuste estadístico de parámetros Cox: %s", exc)
            raise

    def predict_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Predice el tiempo esperado restante de los buques en tránsito.
        """
        X = df[self.covariates].dropna()
        try:
            # conditional_time nos daría el restante dado los que llevan t tiempo vivo
            # El método predict_expectation() nos indica el Expected Time del evento usando AUC interpolado
            preds = self.model.predict_expectation(X)
            return preds
        except Exception as exc:
            logger.error("No se pudo inferir el riesgo: %s", exc)
            raise

    def print_summary(self) -> None:
        """Fuerza a Lifelines a imprimir la tabla de Hazard Ratios y P-values."""
        if hasattr(self.model, "summary"):
            self.model.print_summary()
        else:
            logger.warning("Intentaste imprimir un resumen, pero el modelo no ha sido entrenado.")
