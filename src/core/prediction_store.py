#!/usr/bin/env python3
"""Per-fold prediction vectors, persisted for cross-paradigm comparison.

The framework's central claim is that the paradigms produce identical
predictions, which requires the prediction vectors themselves rather than the
aggregate metrics derived from them: equal R2 is necessary for equal predictions,
not sufficient.

Alignment. Rows arrive already ordered by (entity, year), applied identically by
every paradigm, so position is canonical. Each record additionally carries its
entity and observed target, and the comparison rejects any pair whose entity
sequence or observed targets differ before it looks at the predictions. Two
paradigms disagreeing on which rows they evaluated is a distinct failure from two
paradigms disagreeing on the values they predicted, and conflating them would
report the first as the second.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

PREDICTIONS_SUBDIR = "predictions"
_COLUMNS = ["fold", "model", "row", "entity", "y_true", "y_pred"]


class PredictionRecorder:
    """Accumulates prediction vectors and writes one artifact per paradigm."""

    def __init__(self, paradigm: str):
        self.paradigm = paradigm
        self._rows: List[Dict[str, Any]] = []

    def record(
        self,
        fold: int,
        model: str,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        entities: Optional[Sequence[Any]] = None,
    ) -> None:
        """Store one prediction vector.

        Raises when the vectors disagree in length, which would otherwise
        surface later as an alignment failure attributed to the wrong cause.
        """
        true_values = np.asarray(y_true, dtype=float).ravel()
        predicted = np.asarray(y_pred, dtype=float).ravel()
        if true_values.shape != predicted.shape:
            raise ValueError(
                f"{self.paradigm} fold {fold} model {model!r}: y_true has "
                f"{true_values.shape[0]} values and y_pred has "
                f"{predicted.shape[0]}"
            )

        if entities is None:
            keys: Iterable[Any] = [None] * len(true_values)
        else:
            keys = list(np.asarray(entities, dtype=object).ravel())
            if len(keys) != len(true_values):
                raise ValueError(
                    f"{self.paradigm} fold {fold} model {model!r}: "
                    f"{len(keys)} entities for {len(true_values)} predictions"
                )

        for position, (entity, observed, prediction) in enumerate(
            zip(keys, true_values, predicted)
        ):
            self._rows.append({
                "fold": int(fold),
                "model": str(model),
                "row": position,
                "entity": None if entity is None else str(entity),
                "y_true": float(observed),
                "y_pred": float(prediction),
            })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=_COLUMNS)

    def write(self, path: str) -> Optional[str]:
        """Write the accumulated vectors as Parquet. No-op when empty."""
        if not self._rows:
            return None
        frame = self.frame()
        frame.to_parquet(path, index=False)
        return path


def predictions_path(paradigm: str, stage: str) -> str:
    """Absolute path of one stage's prediction artifact for a paradigm.

    Baseline and hierarchical models run as separate processes, so each writes
    its own file; a single shared path would have the second overwrite the first.
    """
    from core.config import get_absolute_output_path

    return get_absolute_output_path(
        f"ml_pipeline/architectures/{paradigm}/{PREDICTIONS_SUBDIR}/"
        f"predictions_{stage}_{paradigm}.parquet"
    )


def load_predictions(paradigm: str) -> Optional[pd.DataFrame]:
    """Read every stage's predictions for a paradigm, or None when absent."""
    import glob
    import os

    from core.config import get_absolute_output_path

    pattern = get_absolute_output_path(
        f"ml_pipeline/architectures/{paradigm}/{PREDICTIONS_SUBDIR}/"
        f"predictions_*_{paradigm}.parquet"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None

    frames = [pd.read_parquet(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return None

    duplicated = combined.duplicated(subset=["fold", "model", "row"])
    if duplicated.any():
        offending = combined.loc[duplicated, ["fold", "model"]].drop_duplicates()
        raise ValueError(
            f"{paradigm}: prediction artifacts overlap on "
            f"{len(offending)} (fold, model) pair(s); one stage overwrote "
            f"another's vectors"
        )
    return combined
