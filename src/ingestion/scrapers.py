"""
Módulo: scrapers.py
Responsabilidad: Ingesta de datos de fuentes de noticias y eventos globales.
Fuentes objetivo: GDELT Project, Reuters, Financial Times (RSS), UN News.

Principios aplicados:
- SRP: Cada scraper se ocupa de una sola fuente.
- ISP: Interfaz mínima común para todos los scrapers.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import date

import feedparser

from src.ingestion.api_client import APIClient, APIClientError

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Interfaz abstracta para todos los scrapers de datos.

    Define el contrato mínimo que deben cumplir las implementaciones concretas.
    """

    @abstractmethod
    def fetch(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Obtiene registros de la fuente de datos.

        Returns:
            Lista de registros normalizados como diccionarios.

        Raises:
            NotImplementedError: Si la subclase no implementa este método.
        """
        raise NotImplementedError


class GDELTScraper(BaseScraper):
    """
    Scraper para GDELT Project API (eventos geopolíticos globales).

    Attributes:
        client (APIClient): Cliente HTTP configurado para GDELT.
    """

    GDELT_BASE_URL: str = "https://api.gdeltproject.org/api/v2"

    def __init__(self) -> None:
        """Inicializa el scraper con el cliente HTTP de GDELT."""
        self.client = APIClient(base_url=self.GDELT_BASE_URL)

    def fetch(
        self,
        query: str = "supply chain disruption",
        mode: str = "artlist",
        max_records: int = 250,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Consulta artículos de GDELT filtrados por query y rango de fechas.

        Args:
            query: Término de búsqueda para filtrar eventos.
            mode: Modo de la API GDELT (artlist, timelinevol, etc.).
            max_records: Número máximo de registros a recuperar.
            start_date: Fecha de inicio del rango (opcional).
            end_date: Fecha de fin del rango (opcional).

        Returns:
            Lista de artículos/eventos normalizados.

        Raises:
            APIClientError: Si la petición HTTP falla.
        """
        params: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "maxrecords": max_records,
            "format": "json",
        }
        if start_date:
            params["startdatetime"] = start_date.strftime("%Y%m%d000000")
        if end_date:
            params["enddatetime"] = end_date.strftime("%Y%m%d235959")

        try:
            logger.info("Fetching GDELT | query='%s' | max=%d", query, max_records)
            response = self.client.get("/doc/doc", params=params)
            articles: List[Dict[str, Any]] = response.get("articles", [])
            logger.info("GDELT returned %d articles.", len(articles))
            return articles
        except APIClientError as exc:
            logger.error("GDELTScraper.fetch failed: %s", exc)
            raise


class RSSNewsScraper(BaseScraper):
    """
    Scraper genérico para fuentes de noticias via RSS/Atom.

    Attributes:
        feed_url (str): URL del feed RSS a consumir.
        source_name (str): Nombre identificador de la fuente.
    """

    def __init__(self, feed_url: str, source_name: str = "unknown") -> None:
        """
        Inicializa el scraper RSS.

        Args:
            feed_url: URL del feed RSS/Atom.
            source_name: Nombre de la fuente (para trazabilidad).
        """
        self.feed_url = feed_url
        self.source_name = source_name

    def fetch(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Parsea y normaliza las entradas del feed RSS.

        Returns:
            Lista de artículos con campos: title, link, published, summary, source.

        Raises:
            RuntimeError: Si el feed no puede ser parseado.
        """
        logger.info("Fetching RSS feed: %s [%s]", self.feed_url, self.source_name)
        try:
            feed = feedparser.parse(self.feed_url)
            if feed.bozo:
                logger.warning("Feed malformado en %s: %s", self.source_name, feed.bozo_exception)
            entries = [
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", ""),
                    "source": self.source_name,
                }
                for entry in feed.entries
            ]
            logger.info("RSS [%s] returned %d entries.", self.source_name, len(entries))
            return entries
        except Exception as exc:
            logger.error("RSSNewsScraper.fetch failed for %s: %s", self.source_name, exc)
            raise RuntimeError(f"Error parseando feed {self.source_name}") from exc
