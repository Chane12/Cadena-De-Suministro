"""
Módulo: cleaner.py
Responsabilidad: Limpieza, normalización y validación de DataFrames crudos.

Principios aplicados:
- SRP: Solo transformaciones de limpieza.
- Encapsulación: Reglas consolidadas bajo la clase DataCleaner.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataCleaner:
    """Encargado de la limpieza de datos de tráfico logístico y fuentes de inteligencia."""

    def clean_shipping_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia los datos del log de tráfico marítimo.

        - Convierte fechas strings a objetos datetime reales.
        - Asegura que delay_days sea numérico (float).
        - Rellena delay_days vacíos con 0.0 para observaciones parciales/censuradas.
        """
        logger.info("Iniciando clean_shipping_data (%d filas)", len(df))
        df_clean = df.copy()

        # Parsear fechas (coerce convierte los in-transit a NaT si strings vacías o malformados)
        df_clean["departure_date"] = pd.to_datetime(df_clean["departure_date"], errors="coerce")
        df_clean["arrival_date"] = pd.to_datetime(df_clean["arrival_date"], errors="coerce")
        
        # Casting de delay_days y manejo de nulos (aunque el mock generator no emite nulos reales aquí,
        # en producción las APIs pueden mandar missing values).
        df_clean["delay_days"] = pd.to_numeric(df_clean["delay_days"], errors="coerce").fillna(0.0).astype(float)
        
        logger.info("clean_shipping_data: Completado con éxito. Shape final: %s", df_clean.shape)
        return df_clean

    def clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza de texto básica para el feed de noticias.

        - Despliega caracteres alfanuméricos puros en 'headline_clean'
        - Elimina el label de sentimiento falso (mock), dado que en producción
          el backend nos debe dar el scoring a partir de texto base.
        """
        logger.info("Iniciando clean_news_data (%d filas)", len(df))
        df_clean = df.copy()

        # Simulando entorno productivo: las APIs no nos dan "sentiment_labels" curados
        if "sentiment_label" in df_clean.columns:
            logger.info("Eliminando 'sentiment_label' para simular entorno en crudo real.")
            df_clean = df_clean.drop(columns=["sentiment_label"])

        # Estandarización de fecha
        if "date" in df_clean.columns:
            df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
            df_clean = df_clean.dropna(subset=["date"]).copy()

        # Limpieza principal de texto (Regex)
        if "headline" in df_clean.columns:
            # Creamos una columna auxiliar estandarizada
            df_clean["headline_clean"] = df_clean["headline"].str.lower()
            df_clean["headline_clean"] = df_clean["headline_clean"].str.replace(r'[^a-z0-9\s]', '', regex=True)
            df_clean["headline_clean"] = df_clean["headline_clean"].str.strip()
            
            # Quitar aquellos que se hayan quedado vacíos tras limpiar rarezas
            df_clean["headline_clean"] = df_clean["headline_clean"].replace('', np.nan)
            df_clean = df_clean.dropna(subset=["headline_clean"]).copy()

        logger.info("clean_news_data: Completado. Shape final: %s", df_clean.shape)
        return df_clean
