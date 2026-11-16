"""Configuração do Hugging Face antes da importação das bibliotecas do Hub."""
import os
import warnings
from functools import lru_cache
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def configurar_huggingface():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if os.environ.get("HF_HUB_DISABLE_XET", "").upper() in {"1", "ON", "YES", "TRUE"}:
        warnings.warn("Xet desabilitado; downloads usarão HTTP sem aceleração.", RuntimeWarning, stacklevel=2)
        return
    try:
        import_module("hf_xet")
    except ImportError:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        warnings.warn("hf_xet indisponível; downloads usarão HTTP sem aceleração.", RuntimeWarning, stacklevel=2)
