from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from riskrs.typing import IntoExprColumn

LIB = Path(__file__).parent


def auc(y_true: IntoExprColumn, y_score: IntoExprColumn) -> pl.Expr:
    """
    Calculate the Area Under the Curve (AUC) for binary classification.

    Parameters
    ----------
    y_true : IntoExprColumn
        True binary labels.
    y_score : IntoExprColumn
        Target scores, can either be probability estimates of the positive class,
        confidence values, or binary decisions.

    Returns
    -------
    pl.Expr
        A Polars expression representing the AUC.
    """
    return register_plugin_function(
        args=(y_true, y_score),
        plugin_path=LIB,
        function_name="auc",
        is_elementwise=False,
        returns_scalar=True,
    )


__all__: list[str] = []
