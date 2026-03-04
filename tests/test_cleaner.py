"""
Tests unitarios para el módulo de limpieza de datos (src/processing/cleaner.py).
"""

import pytest
import pandas as pd
import numpy as np

from src.processing.cleaner import (
    remove_duplicates,
    handle_missing_values,
    normalize_column_names,
    cast_dtypes,
    DataCleaningError,
)


class TestRemoveDuplicates:
    """Tests para la función remove_duplicates."""

    def test_removes_exact_duplicates(self) -> None:
        """Debe eliminar filas completamente duplicadas."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self) -> None:
        """DataFrame sin duplicados no debe cambiar."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = remove_duplicates(df)
        assert len(result) == 3

    def test_subset_duplicates(self) -> None:
        """Solo debe considerar el subconjunto de columnas especificado."""
        df = pd.DataFrame({"a": [1, 1], "b": [10, 20]})
        result = remove_duplicates(df, subset=["a"])
        assert len(result) == 1


class TestHandleMissingValues:
    """Tests para la función handle_missing_values."""

    def test_drop_strategy(self) -> None:
        """Strategy 'drop' debe eliminar filas con NaN."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = handle_missing_values(df, strategy="drop")
        assert len(result) == 2
        assert result["a"].isna().sum() == 0

    def test_fill_strategy(self) -> None:
        """Strategy 'fill' debe rellenar NaN con el valor especificado."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = handle_missing_values(df, strategy="fill", fill_value=0.0)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == 0.0

    def test_fill_strategy_without_value_raises(self) -> None:
        """'fill' sin fill_value debe lanzar DataCleaningError."""
        df = pd.DataFrame({"a": [1.0, np.nan]})
        with pytest.raises(DataCleaningError):
            handle_missing_values(df, strategy="fill")

    def test_mean_strategy(self) -> None:
        """Strategy 'mean' debe rellenar con la media de la columna."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = handle_missing_values(df, strategy="mean")
        assert result["a"].iloc[1] == pytest.approx(2.0)

    def test_invalid_strategy_raises(self) -> None:
        """Estrategia desconocida debe lanzar DataCleaningError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(DataCleaningError, match="Estrategia desconocida"):
            handle_missing_values(df, strategy="invalid_strategy")


class TestNormalizeColumnNames:
    """Tests para la función normalize_column_names."""

    def test_lowercase_conversion(self) -> None:
        """Debe convertir nombres a minúsculas."""
        df = pd.DataFrame(columns=["Name", "VALUE", "Index"])
        result = normalize_column_names(df)
        assert list(result.columns) == ["name", "value", "index"]

    def test_spaces_to_underscores(self) -> None:
        """Debe reemplazar espacios por guiones bajos."""
        df = pd.DataFrame(columns=["Column Name", "Another Col"])
        result = normalize_column_names(df)
        assert "column_name" in result.columns
        assert "another_col" in result.columns


class TestCastDtypes:
    """Tests para la función cast_dtypes."""

    def test_cast_to_int(self) -> None:
        """Debe castear columna a entero."""
        df = pd.DataFrame({"a": ["1", "2", "3"]})
        result = cast_dtypes(df, {"a": "int64"})
        assert result["a"].dtype == np.int64

    def test_cast_to_datetime(self) -> None:
        """Debe castear columna a datetime."""
        df = pd.DataFrame({"fecha": ["2024-01-01", "2024-06-15"]})
        result = cast_dtypes(df, {"fecha": "datetime64[ns]"})
        assert pd.api.types.is_datetime64_any_dtype(result["fecha"])

    def test_missing_column_warns_no_error(self) -> None:
        """Columna inexistente debe logear warning pero no lanzar excepción."""
        df = pd.DataFrame({"a": [1, 2]})
        result = cast_dtypes(df, {"nonexistent": "int64"})
        assert "nonexistent" not in result.columns
