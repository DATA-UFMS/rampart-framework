"""Configuração Pytest: adiciona src/ ao sys.path para que imports 'from core...' funcionem."""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
