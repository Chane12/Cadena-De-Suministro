"""
Módulo: risk_index.py
Responsabilidad: Cálculo del Global Supply Chain Resilience Index (GSCRI).

El índice agrega múltiples dimensiones de riesgo ponderadas:
  - Riesgo geopolítico (fuente: GDELT / GPR Index)
  - Riesgo climático (fuente: EM-DAT / NOAA)
  - Concentración de proveedores (Índice Herfindahl-Hirschman)
  - Conectividad logística (fuente: World Bank LPI)
  - Sentimiento de prensa (output NLP Pipeline)

Fórmula base: GSCRI = Σ (wᵢ × zᵢ), donde zᵢ = z-score de cada dimensión.

Principios aplicados:
- SRP: Solo cálculo del índice.
- OCP: Nuevas dimensiones se añaden sin modificar el núcleo.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskDimension:
    """
    Representa una dimensión de riesgo con su peso relativo.

    Attributes:
        name: Identificador de la dimensión.
        column: Nombre de la columna en el DataFrame de entrada.
        weight: Peso en el índice compuesto (deben sumar 1.0 en total).
        higher_is_riskier: Si True, valores altos = mayor riesgo. Si False, se invierte.
    """
    name: str
    column: str
    weight: float
    higher_is_riskier: bool = True


DEFAULT_DIMENSIONS: List[RiskDimension] = [
    RiskDimension("Geopolitical Risk",   "geopolitical_risk_score",   weight=0.30),
    RiskDimension("Climate Risk",        "climate_risk_score",         weight=0.25),
    RiskDimension("Supplier Concentration", "hhi_index",              weight=0.20),
    RiskDimension("Logistics Connectivity", "lpi_score",              weight=0.15, higher_is_riskier=False),
    RiskDimension("Press Sentiment",     "negative_sentiment_score",   weight=0.10),
]


class ResilienceIndexCalculator:
    """
    Calcula el Global Supply Chain Resilience Index (GSCRI) para un conjunto de rutas/países.

    El proceso es:
      1. Z-score de cada dimensión (normalización).
      2. Inversión de dimensiones donde mayor valor = menor riesgo.
      3. Promedio ponderado.
      4. Rescalado a [0, 100] donde 100 = máximo riesgo.

    Attributes:
        dimensions (List[RiskDimension]): Dimensiones de riesgo y sus pesos.
    """

    INDEX_COLUMN: str = "gscri_score"

    def __init__(self, dimensions: List[RiskDimension] = None) -> None:
        """
        Inicializa el calculador con las dimensiones configuradas.

        Args:
            dimensions: Lista de RiskDimension. Si None, usa DEFAULT_DIMENSIONS.

        Raises:
            ValueError: Si los pesos de las dimensiones no suman aproximadamente 1.0.
        """
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        total_weight = sum(d.weight for d in self.dimensions)
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"Los pesos deben sumar 1.0, pero suman {total_weight:.4f}.")
        logger.info("ResilienceIndexCalculator inicializado con %d dimensiones.", len(self.dimensions))

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el GSCRI para cada fila del DataFrame.

        Args:
            df: DataFrame con las columnas requeridas por cada RiskDimension.

        Returns:
            DataFrame original con la columna 'gscri_score' añadida (0-100).

        Raises:
            KeyError: Si alguna columna requerida falta en el DataFrame.
            ValueError: Si el DataFrame está vacío.
        """
        if df.empty:
            raise ValueError("El DataFrame de entrada está vacío.")

        missing_cols = [d.column for d in self.dimensions if d.column not in df.columns]
        if missing_cols:
            raise KeyError(f"Columnas faltantes en el DataFrame: {missing_cols}")

        result = df.copy()
        weighted_scores = pd.Series(0.0, index=result.index)

        for dim in self.dimensions:
            col_data = result[dim.column].astype(float)
            z_score = (col_data - col_data.mean()) / (col_data.std() + 1e-9)

            if not dim.higher_is_riskier:
                z_score = -z_score

            weighted_scores += dim.weight * z_score
            logger.debug("Dimensión '%s' aplicada (weight=%.2f).", dim.name, dim.weight)

        # Rescalar a [0, 100]
        min_score, max_score = weighted_scores.min(), weighted_scores.max()
        if np.isclose(min_score, max_score):
            result[self.INDEX_COLUMN] = 50.0
        else:
            result[self.INDEX_COLUMN] = (
                (weighted_scores - min_score) / (max_score - min_score) * 100
            ).round(2)

        logger.info(
            "GSCRI calculado. Media=%.2f | Std=%.2f | Min=%.2f | Max=%.2f",
            result[self.INDEX_COLUMN].mean(),
            result[self.INDEX_COLUMN].std(),
            result[self.INDEX_COLUMN].min(),
            result[self.INDEX_COLUMN].max(),
        )
        return result
