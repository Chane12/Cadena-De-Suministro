"""
Tests unitarios para el cálculo del índice GSCRI (src/models/risk_index.py).
"""

import pytest
import pandas as pd
import numpy as np

from src.models.risk_index import ResilienceIndexCalculator, RiskDimension, DEFAULT_DIMENSIONS


def _make_sample_df(n: int = 10) -> pd.DataFrame:
    """Genera un DataFrame de prueba con las columnas del GSCRI."""
    np.random.seed(42)
    return pd.DataFrame({
        "country": [f"COUNTRY_{i}" for i in range(n)],
        "geopolitical_risk_score": np.random.uniform(0, 100, n),
        "climate_risk_score": np.random.uniform(0, 100, n),
        "hhi_index": np.random.uniform(0, 1, n),
        "lpi_score": np.random.uniform(1, 5, n),
        "negative_sentiment_score": np.random.uniform(0, 1, n),
    })


class TestResilienceIndexCalculator:
    """Tests para ResilienceIndexCalculator."""

    def test_output_column_exists(self) -> None:
        """El DataFrame resultado debe contener la columna 'gscri_score'."""
        calc = ResilienceIndexCalculator()
        df = _make_sample_df()
        result = calc.compute(df)
        assert "gscri_score" in result.columns

    def test_scores_in_range(self) -> None:
        """Los scores GSCRI deben estar en el rango [0, 100]."""
        calc = ResilienceIndexCalculator()
        df = _make_sample_df(20)
        result = calc.compute(df)
        assert result["gscri_score"].min() >= 0.0
        assert result["gscri_score"].max() <= 100.0

    def test_empty_dataframe_raises(self) -> None:
        """DataFrame vacío debe lanzar ValueError."""
        calc = ResilienceIndexCalculator()
        with pytest.raises(ValueError, match="vacío"):
            calc.compute(pd.DataFrame())

    def test_missing_column_raises(self) -> None:
        """DataFrame sin columnas requeridas debe lanzar KeyError."""
        calc = ResilienceIndexCalculator()
        df = pd.DataFrame({"irrelevant_col": [1, 2, 3]})
        with pytest.raises(KeyError):
            calc.compute(df)

    def test_invalid_weights_raises(self) -> None:
        """Pesos que no suman 1.0 deben lanzar ValueError."""
        bad_dims = [
            RiskDimension("A", "col_a", weight=0.5),
            RiskDimension("B", "col_b", weight=0.3),  # Total = 0.8
        ]
        with pytest.raises(ValueError, match="pesos"):
            ResilienceIndexCalculator(dimensions=bad_dims)

    def test_preserves_original_columns(self) -> None:
        """El resultado debe contener todas las columnas originales."""
        calc = ResilienceIndexCalculator()
        df = _make_sample_df()
        result = calc.compute(df)
        for col in df.columns:
            assert col in result.columns
