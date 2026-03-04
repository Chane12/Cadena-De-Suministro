"""
Módulo: mock_generator.py
Responsabilidad: Generación de datos sintéticos realistas para el pipeline
del Global Supply Chain Resilience Index (GSCRI).

Desacopla el desarrollo de la dependencia de APIs externas, permitiendo
probar los modelos de supervivencia y el pipeline de limpieza sin conexión.

Principios aplicados:
- SRP: Solo genera datos simulados, sin lógica de negocio ni persistencia.
- OCP: Nuevos generadores se añaden como métodos públicos sin modificar los existentes.
- DRY: Plantillas de titulares y rutas centralizadas en constantes de clase.

Nota sobre Censura (Survival Analysis):
  En análisis de supervivencia, un evento "censurado" (event_observed=0) significa
  que el barco aún no ha llegado durante el período de observación, por lo que no
  conocemos el delay final. Un evento observado (event_observed=1) indica que el
  barco llegó (con o sin retraso) y el delay_days es el valor real conocido.
"""

import logging
import random
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)


class MockDataGenerator:
    """
    Generador de datos sintéticos para rutas marítimas y noticias de supply chain.

    Produce DataFrames estadísticamente consistentes con las propiedades
    requeridas por los modelos de supervivencia y el pipeline NLP downstream.

    Attributes:
        faker (Faker): Instancia de Faker para generación de texto realista.
        rng (np.random.Generator): Generador de números aleatorios reproducible.
        seed (int): Semilla de aleatoriedad para reproducibilidad.
    """

    # ─── Constantes de Dominio ────────────────────────────────────────────────

    TRADE_ROUTES: List[str] = [
        "Shanghai-Rotterdam",
        "Shenzhen-Los Angeles",
        "Singapore-Hamburg",
        "Busan-Long Beach",
        "Guangzhou-Antwerp",
        "Hong Kong-Felixstowe",
        "Ningbo-Valencia",
        "Tianjin-Savannah",
        "Port Klang-Rotterdam",
        "Colombo-Hamburg",
        "Dubai-Barcelona",
        "Mumbai-Jeddah",
        "Callao-Rotterdam",
        "Santos-Algeciras",
        "Durban-Antwerp",
    ]

    VESSEL_STATUS: List[str] = ["Arrived", "In Transit", "Delayed"]
    STATUS_WEIGHTS: List[float] = [0.55, 0.25, 0.20]  # distribución realista

    # Plantillas de titulares por categoría de riesgo
    NOISE_HEADLINES: List[str] = [
        "Port operations normal in {port}",
        "Cargo volumes at {port} remain stable this quarter",
        "Shipping lines report smooth operations through {strait}",
        "Container throughput increases at {port} year-on-year",
        "Fuel costs stabilize for {port} routes",
        "New berth inaugurated at {port} to boost capacity",
        "Logistics efficiency improves at {region} terminals",
    ]

    POLITICAL_RISK_HEADLINES: List[str] = [
        "New tariffs announced between {country_a} and {country_b}",
        "Trade war escalation: {country_a} imposes sanctions on {country_b}",
        "Port workers strike threatens supply chain in {region}",
        "Geopolitical tensions disrupt {strait} shipping lanes",
        "Military exercises near {strait} raise shipping concerns",
        "{country_a} bans exports of critical minerals to {country_b}",
        "Diplomatic crisis between {country_a} and {country_b} halts trade talks",
        "New customs regulations slow {region} imports significantly",
    ]

    DISASTER_HEADLINES: List[str] = [
        "Typhoon warning near {strait}",
        "Earthquake disrupts port operations in {region}",
        "Severe flooding closes {port} for 72 hours",
        "Cyclone forces vessel rerouting around {region}",
        "Fog advisory issued for {strait}: delays expected",
        "Wildfire threatens industrial zones near {port}",
        "Drought reduces {region} river transport capacity",
        "Tsunami warning affects {port}: evacuations underway",
    ]

    # Datos geográficos de relleno para plantillas
    _PORTS: List[str] = [
        "Rotterdam", "Shanghai", "Singapore", "Los Angeles", "Antwerp",
        "Hamburg", "Busan", "Long Beach", "Valencia", "Felixstowe",
    ]
    _STRAITS: List[str] = [
        "Taiwan Strait", "Strait of Hormuz", "Suez Canal", "Panama Canal",
        "Strait of Malacca", "English Channel", "Bosphorus",
    ]
    _REGIONS: List[str] = [
        "Southeast Asia", "Middle East", "East Africa", "South America",
        "Northern Europe", "West Africa", "Indian Subcontinent", "East China",
    ]
    _COUNTRIES: List[str] = [
        "US", "China", "EU", "Russia", "Iran", "South Korea",
        "Japan", "India", "Brazil", "Turkey",
    ]
    _SOURCES: List[str] = [
        "Reuters", "Bloomberg", "Financial Times", "Lloyd's List",
        "TradeWinds", "Splash247", "JOC.com", "Freightos Media",
    ]

    def __init__(self, seed: int = 42, locale: str = "en_US") -> None:
        """
        Inicializa el generador con una semilla fija para reproducibilidad.

        Args:
            seed: Semilla para numpy y Faker. Garantiza datasets reproducibles.
            locale: Locale de Faker para generación de nombres y texto.
        """
        self.seed = seed
        self.faker = Faker(locale)
        Faker.seed(seed)
        self.rng = np.random.default_rng(seed)
        logger.info("MockDataGenerator inicializado con seed=%d, locale='%s'.", seed, locale)

    # ─── Métodos Privados de Apoyo ─────────────────────────────────────────────

    def _generate_vessel_id(self) -> str:
        """Genera un ID de barco realista formato IMO-XXXXXXX."""
        return f"IMO-{self.rng.integers(1_000_000, 9_999_999)}"

    def _fill_template(self, template: str) -> str:
        """
        Rellena una plantilla de titular con datos geográficos aleatorios.

        Args:
            template: String con placeholders {port}, {strait}, {region},
                      {country_a}, {country_b}.

        Returns:
            Titular con placeholders sustituidos por valores concretos.
        """
        country_pair = self.rng.choice(self._COUNTRIES, size=2, replace=False)
        return template.format(
            port=self.rng.choice(self._PORTS),
            strait=self.rng.choice(self._STRAITS),
            region=self.rng.choice(self._REGIONS),
            country_a=country_pair[0],
            country_b=country_pair[1],
        )

    def _sample_delay_days(self, status: str) -> float:
        """
        Muestrea el valor de delay_days según el estado del barco.

        Distribuciones:
          - 'Arrived':    Weibull(k=1.5, λ=3) → retrasos cortos, mayoría sin retraso.
          - 'Delayed':    Weibull(k=2.0, λ=12) → retrasos moderados-severos.
          - 'In Transit': Parcial uniforme [0, duration_observable] → censurado.

        Args:
            status: Estado del barco ('Arrived', 'Delayed', 'In Transit').

        Returns:
            Número de días (float, redondeado a 1 decimal).
        """
        if status == "Arrived":
            # Llegó: el retraso es pequeño o nulo
            raw = self.rng.weibull(1.5) * 3.0
            return round(max(0.0, raw), 1)
        elif status == "Delayed":
            # Retrasado: retraso más severo
            raw = self.rng.weibull(2.0) * 12.0
            return round(max(1.0, raw), 1)
        else:
            # En tránsito: observamos tiempo parcial (valor censurado)
            return round(self.rng.uniform(1.0, 20.0), 1)

    # ─── Métodos Públicos ──────────────────────────────────────────────────────

    def generate_shipping_log(self, n_vessels: int = 1000) -> pd.DataFrame:
        """
        Genera un log sintético de tráfico marítimo con semántica de Survival Analysis.

        Columnas del DataFrame resultante:
          - vessel_id (str):        ID único del barco formato 'IMO-XXXXXXX'.
          - route_id (str):         Ruta origen-destino (ej. 'Shanghai-Rotterdam').
          - departure_date (date):  Fecha de salida del puerto de origen.
          - arrival_date (date):    Fecha estimada de llegada (puede ser nula si 'In Transit').
          - status (str):           'Arrived' | 'In Transit' | 'Delayed'.
          - delay_days (float):     Días de retraso observados o parciales (censurado).
          - event_observed (int):   1 = evento ocurrido (llegó); 0 = censurado (aún en tránsito).
          - cargo_type (str):       Tipo de carga a bordo.
          - vessel_capacity_teu (int): Capacidad en TEUs (Twenty-foot Equivalent Units).
          - geopolitical_risk (float): Score de riesgo en la ruta [0, 1].
          - weather_severity (float):  Score de severidad climática [0, 1].

        Args:
            n_vessels: Número de registros a generar. Default=1000.

        Returns:
            DataFrame con n_vessels filas y las columnas descritas arriba.

        Raises:
            ValueError: Si n_vessels es menor o igual a 0.
        """
        if n_vessels <= 0:
            raise ValueError(f"n_vessels debe ser > 0. Recibido: {n_vessels}")

        logger.info("Generando shipping log con %d registros...", n_vessels)

        # Muestrear estados con distribución realista
        statuses = self.rng.choice(
            self.VESSEL_STATUS,
            size=n_vessels,
            p=self.STATUS_WEIGHTS,
        ).tolist()

        cargo_types = ["Electronics", "Automotive", "Food & Beverage", "Chemicals",
                       "Textiles", "Machinery", "Raw Materials", "Pharmaceuticals"]

        records: List[Dict[str, Any]] = []
        for i in range(n_vessels):
            status = statuses[i]
            departure = self.faker.date_between(
                start_date=date(2022, 1, 1),
                end_date=date(2024, 12, 31),
            )
            delay = self._sample_delay_days(status)

            # Lógica de censura para Survival Analysis
            # event_observed=1 → el barco llegó (evento real conocido)
            # event_observed=0 → aún en tránsito (dato censurado a la derecha)
            event_observed = 0 if status == "In Transit" else 1

            # arrival_date solo disponible si el barco ya llegó
            if status in ("Arrived", "Delayed"):
                scheduled_days = int(self.rng.integers(14, 45))
                arrival_date = departure + timedelta(days=scheduled_days + int(delay))
            else:
                arrival_date = None

            records.append({
                "vessel_id":            self._generate_vessel_id(),
                "route_id":             str(self.rng.choice(self.TRADE_ROUTES)),
                "departure_date":       departure,
                "arrival_date":         arrival_date,
                "status":               status,
                "delay_days":           delay,
                "event_observed":       event_observed,
                "cargo_type":           str(self.rng.choice(cargo_types)),
                "vessel_capacity_teu":  int(self.rng.integers(500, 24_000)),
                "geopolitical_risk":    round(float(self.rng.beta(2, 5)), 4),
                "weather_severity":     round(float(self.rng.beta(1.5, 6)), 4),
            })

        df = pd.DataFrame(records)
        logger.info(
            "Shipping log generado: %d filas | Arrived=%.1f%% | In Transit=%.1f%% | Delayed=%.1f%%",
            len(df),
            (df["status"] == "Arrived").mean() * 100,
            (df["status"] == "In Transit").mean() * 100,
            (df["status"] == "Delayed").mean() * 100,
        )
        return df

    def generate_news_feed(self, n_news: int = 200) -> pd.DataFrame:
        """
        Genera un feed sintético de noticias de supply chain con tres categorías de riesgo.

        Las noticias se distribuyen en tres categorías con proporciones configuradas:
          - Ruido operacional (50%): operaciones normales, sin señal de riesgo.
          - Riesgo político (30%):   tarifas, sanciones, huelgas, tensiones geopolíticas.
          - Desastres naturales (20%): tifones, terremotos, inundaciones, sequías.

        Columnas del DataFrame resultante:
          - date (date):        Fecha de publicación.
          - headline (str):     Titular de la noticia.
          - source (str):       Medio de comunicación (Reuters, Bloomberg, etc.).
          - region (str):       Región geográfica afectada.
          - risk_category (str): 'noise' | 'political_risk' | 'disaster'.
          - sentiment_label (str): Etiqueta heurística 'positive'|'neutral'|'negative'.

        Args:
            n_news: Número de artículos a generar. Default=200.

        Returns:
            DataFrame con n_news filas.

        Raises:
            ValueError: Si n_news es menor o igual a 0.
        """
        if n_news <= 0:
            raise ValueError(f"n_news debe ser > 0. Recibido: {n_news}")

        logger.info("Generando news feed con %d artículos...", n_news)

        # Distribución de categorías
        category_weights = [0.50, 0.30, 0.20]
        categories = ["noise", "political_risk", "disaster"]
        category_map = {
            "noise":          (self.NOISE_HEADLINES,         "neutral"),
            "political_risk": (self.POLITICAL_RISK_HEADLINES, "negative"),
            "disaster":       (self.DISASTER_HEADLINES,       "negative"),
        }

        records: List[Dict[str, Any]] = []
        for _ in range(n_news):
            category = str(self.rng.choice(categories, p=category_weights))
            templates, default_sentiment = category_map[category]
            template = str(self.rng.choice(templates))
            headline = self._fill_template(template)

            # Variación de sentimiento: el ruido tiene ligera positividad
            if category == "noise":
                sentiment = self.rng.choice(
                    ["positive", "neutral", "negative"],
                    p=[0.30, 0.60, 0.10],
                )
            else:
                sentiment = self.rng.choice(
                    ["positive", "neutral", "negative"],
                    p=[0.05, 0.20, 0.75],
                )

            records.append({
                "date":             self.faker.date_between(
                    start_date=date(2022, 1, 1),
                    end_date=date(2024, 12, 31),
                ),
                "headline":         headline,
                "source":           str(self.rng.choice(self._SOURCES)),
                "region":           str(self.rng.choice(self._REGIONS)),
                "risk_category":    category,
                "sentiment_label":  str(sentiment),
            })

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        logger.info(
            "News feed generado: %d artículos | noise=%.1f%% | political=%.1f%% | disaster=%.1f%%",
            len(df),
            (df["risk_category"] == "noise").mean() * 100,
            (df["risk_category"] == "political_risk").mean() * 100,
            (df["risk_category"] == "disaster").mean() * 100,
        )
        return df
