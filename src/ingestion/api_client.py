"""
Módulo: api_client.py
Responsabilidad: Cliente HTTP genérico y reutilizable para la ingesta de datos
desde APIs externas (World Bank, UN Comtrade, GDELT, etc.).

Principios aplicados:
- SRP: Solo se ocupa de la comunicación HTTP.
- OCP: Extensible mediante subclases sin modificar esta base.
"""

import logging
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Excepción base para errores del cliente API."""
    pass


class APIClient:
    """
    Cliente HTTP base con reintentos, timeout y logging estructurado.

    Attributes:
        base_url (str): URL base de la API destino.
        timeout (int): Segundos máximos de espera por respuesta.
        session (requests.Session): Sesión HTTP con política de reintentos.
    """

    DEFAULT_RETRY_TOTAL: int = 3
    DEFAULT_BACKOFF_FACTOR: float = 1.5
    DEFAULT_TIMEOUT: int = 30

    def __init__(
        self,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Inicializa el cliente HTTP con sesión persistente y política de reintentos.

        Args:
            base_url: URL raíz de la API.
            timeout: Tiempo máximo de espera en segundos.
            headers: Cabeceras HTTP adicionales (ej. API keys).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = self._build_session(headers or {})

    def _build_session(self, headers: Dict[str, str]) -> requests.Session:
        """
        Construye una sesión requests con política de reintentos exponenciales.

        Args:
            headers: Cabeceras a incluir en cada petición.

        Returns:
            Sesión HTTP configurada.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=self.DEFAULT_RETRY_TOTAL,
            backoff_factor=self.DEFAULT_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(headers)
        return session

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Realiza una petición HTTP GET al endpoint especificado.

        Args:
            endpoint: Ruta relativa al base_url.
            params: Parámetros de query string.

        Returns:
            Respuesta JSON deserializada como diccionario.

        Raises:
            APIClientError: Si la petición falla tras los reintentos.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.info("GET %s | params=%s", url, params)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error %s – %s", exc.response.status_code, url)
            raise APIClientError(f"HTTP {exc.response.status_code} en {url}") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error al acceder a %s: %s", url, exc)
            raise APIClientError(f"Error de conexión: {url}") from exc
        except requests.exceptions.Timeout as exc:
            logger.error("Timeout (%ss) en %s", self.timeout, url)
            raise APIClientError(f"Timeout en {url}") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("Error inesperado en GET %s: %s", url, exc)
            raise APIClientError(str(exc)) from exc
