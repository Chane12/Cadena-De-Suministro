"""
Módulo: feature_engineering.py
Responsabilidad: Crear la Master Table combinando datos transaccionales con NLP.
Aplica lógicas de cruce complejas (ventana temporal y pesos por localización) 
para derivar características agregadas como el Voyage Risk Index.

Principios aplicados:
- SRP: Lógica de Feature Engineering separada del Pipeline NLP y del Modelado.
- Vectorización/Optimización donde es posible.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Clase dedicada a crear features derivadas sofisticadas para Machine Learning."""

    def merge_datasets(self, shipping_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el Voyage Risk Index cruzando los intervalos de viaje con las noticias.

        Lógica:
        - Determina la fecha de salida y llegada real o la actual (para censurados).
        - Filtra noticias que ocurrieron durante ese período de tránsito.
        - Calcula el riesgo promedio pesado.
        - Aplica multiplicador (x2.0) si hay coincidencia de región reportada en el titular
          respecto a la ruta del barco.
        """
        logger.info("Iniciando generación del Voyage Risk Index (cruzando %d viajes x %d noticias)...", len(shipping_df), len(news_df))

        # Asegurar formato datetime para operaciones seguras
        shipping_df["departure_date"] = pd.to_datetime(shipping_df["departure_date"], errors="coerce")
        shipping_df["arrival_date"] = pd.to_datetime(shipping_df["arrival_date"], errors="coerce")
        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")

        # Fallback date para barcos "In Transit" (datos censurados a la derecha)
        # Usamos la fecha máxima real de las noticias o la actual
        max_news_date = news_df["date"].max()
        fallback_date = pd.Timestamp.now() if pd.isna(max_news_date) else max_news_date

        voyage_risk_indices = []

        # Iteración barco por barco
        for _, vessel in shipping_df.iterrows():
            start_date = vessel["departure_date"]
            end_date = vessel["arrival_date"]
            
            # Reemplazar NaT con límite superior (el evento aún no ha concluido)
            if pd.isna(end_date):
                end_date = fallback_date

            route = str(vessel.get("route_id", "")).lower()

            # Máscara temporal: Noticias durante este viaje en concreto
            mask = (news_df["date"] >= start_date) & (news_df["date"] <= end_date)
            relevant_news = news_df[mask]

            # Si no hubo noticias en esa ventana
            if relevant_news.empty:
                voyage_risk_indices.append(0.0)
                continue

            weighted_scores = []
            
            # Evaluación del impacto a nivel ruta y agregación
            for _, news in relevant_news.iterrows():
                locs_text = str(news.get("extracted_locations", "unknown")).lower()
                risk = float(news.get("risk_score", 0.0))

                # Weighting System: Bonus Senior.
                weight = 1.0
                if locs_text != "unknown" and locs_text:
                    # Fuzzy validation (contains)
                    loc_list = [l.strip() for l in locs_text.split(",")]
                    # Comprobaremos si alguna locación de la noticia está en la string de 'route'
                    # ej: 'Panama' se encuentra en 'Panama-Los Angeles'
                    for loc in loc_list:
                        if loc and loc in route:
                            weight = 2.0  # El riesgo impacta de frente esta ruta (Magnificador regional)
                            break
                            
                weighted_scores.append(risk * weight)

            # Agregar
            avg_risk = float(np.mean(weighted_scores))
            voyage_risk_indices.append(avg_risk)

        df_master = shipping_df.copy()
        df_master["voyage_risk_index"] = voyage_risk_indices
        
        # Opcional pero recomendado: redondear para evitar artifacts flotantes
        df_master["voyage_risk_index"] = df_master["voyage_risk_index"].round(4)
        
        avg_overall = df_master["voyage_risk_index"].mean()
        max_overall = df_master["voyage_risk_index"].max()
        logger.info("✔️ Voyage Risk Index completado. Riesgo medio de rutas: %.4f | Máximo: %.4f", avg_overall, max_overall)
        
        return df_master
