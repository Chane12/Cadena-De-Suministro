"""
Script: generate_mocks.py
Propósito: Script de ejecución único para generar y persistir los datasets
           sintéticos del proyecto GSCRI en data/raw/.

Uso:
    python generate_mocks.py

Salidas:
    - data/raw/shipping_log.csv   → 1000 registros de tráfico marítimo
    - data/raw/news_feed.csv      → 200 artículos de noticias sintéticas

Este script no pertenece al pipeline de producción; es una utilidad de desarrollo
para desacoplar el entorno de APIs externas durante las fases iniciales.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# ─── Setup de paths para importar src/ ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.mock_generator import MockDataGenerator

# ─── Configuración de Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def save_dataframe(df: pd.DataFrame, path: Path, description: str) -> None:
    """
    Persiste un DataFrame en formato CSV con logging estructurado.

    Args:
        df: DataFrame a guardar.
        path: Ruta de destino del archivo CSV.
        description: Descripción del dataset para el log.

    Raises:
        OSError: Si el directorio no puede crearse o el CSV no puede escribirse.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("✅ %s guardado en: %s (%d filas, %d cols)", description, path, *df.shape)
    except OSError as exc:
        logger.error("❌ Error guardando %s en %s: %s", description, path, exc)
        raise


def display_preview(df: pd.DataFrame, title: str, n_rows: int = 5) -> None:
    """
    Imprime una vista previa del DataFrame con formato tabular.

    Args:
        df: DataFrame a mostrar.
        title: Título descriptivo del dataset.
        n_rows: Número de filas a mostrar. Default=5.
    """
    separator = "═" * 80
    print(f"\n{separator}")
    print(f"  📊  {title}  ({len(df):,} filas × {len(df.columns)} columnas)")
    print(separator)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.head(n_rows).to_string(index=True))
    print(f"\n  Columnas: {df.dtypes.to_dict()}")
    print(separator)


def main() -> None:
    """
    Punto de entrada principal del script de generación de mocks.

    Genera los datasets sintéticos, los guarda en data/raw/ y
    muestra una vista previa de las primeras 5 filas de cada uno.
    """
    logger.info("=" * 60)
    logger.info("  GSCRI – Generador de Datos Sintéticos")
    logger.info("=" * 60)

    generator = MockDataGenerator(seed=42)

    # ── 1. Shipping Log ────────────────────────────────────────────────────────
    logger.info("Generando Shipping Log (n=1000)...")
    shipping_df = generator.generate_shipping_log(n_vessels=1000)
    save_dataframe(
        df=shipping_df,
        path=PROJECT_ROOT / "data" / "raw" / "shipping_log.csv",
        description="Shipping Log",
    )
    display_preview(shipping_df, "SHIPPING LOG – Primeras 5 filas")

    # ── 2. News Feed ───────────────────────────────────────────────────────────
    logger.info("Generando News Feed (n=200)...")
    news_df = generator.generate_news_feed(n_news=200)
    save_dataframe(
        df=news_df,
        path=PROJECT_ROOT / "data" / "raw" / "news_feed.csv",
        description="News Feed",
    )
    display_preview(news_df, "NEWS FEED – Primeras 5 filas")

    # ── 3. Estadísticas de Resumen ─────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  📈  ESTADÍSTICAS DE CENSURA (Survival Analysis)")
    print("═" * 80)
    total = len(shipping_df)
    censored = (shipping_df["event_observed"] == 0).sum()
    observed = (shipping_df["event_observed"] == 1).sum()
    print(f"  Total registros : {total:,}")
    print(f"  Eventos reales  : {observed:,}  ({observed/total*100:.1f}%)")
    print(f"  Censurados      : {censored:,}  ({censored/total*100:.1f}%)")
    print(f"  Delay medio     : {shipping_df['delay_days'].mean():.2f} días")
    print(f"  Delay máximo    : {shipping_df['delay_days'].max():.2f} días")
    print("═" * 80)

    print("\n" + "═" * 80)
    print("  📰  DISTRIBUCIÓN DE CATEGORÍAS (News Feed)")
    print("═" * 80)
    dist = news_df["risk_category"].value_counts()
    for cat, count in dist.items():
        print(f"  {cat:<20}: {count:>3}  ({count/len(news_df)*100:.1f}%)")
    print("═" * 80)

    logger.info("✅ Generación completada. Archivos disponibles en data/raw/.")


if __name__ == "__main__":
    main()
