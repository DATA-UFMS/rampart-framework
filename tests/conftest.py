"""Pytest configuration: adds src/ to sys.path so that 'from core...' imports work."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def audit_panel(panel, features, target_column='target', *,
                unaudited=None, autoregressive=None, config=None):
    """Run the P3 re-audit over a panel expressed as columns plus names.

    `audit_feature_set` takes the fitted design matrix and its target, which is
    what removed the scope ambiguity that had the three paradigms auditing
    different frames. Tests, though, are naturally written as a panel plus a
    list of column names, so this adapts one to the other in one place instead
    of nineteen.

    `unaudited` defaults to every feature: a test that builds a panel to see a
    check fire wants the check to consider what it built. Production passes the
    columns selection never saw, which today is empty.
    """
    from core.scientific_config import SCIENTIFIC_CONFIG
    from core.validation import audit_feature_set

    features = list(features)
    if autoregressive is None:
        autoregressive = [f for f in features if '_lag_' in f]
    return audit_feature_set(
        panel[features], panel[target_column],
        autoregressive=autoregressive,
        unaudited_by_selection=(features if unaudited is None else unaudited),
        config=SCIENTIFIC_CONFIG if config is None else config)
