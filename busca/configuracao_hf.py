"""Compatibilidade para execução direta dos scripts de busca."""
import sys
from pathlib import Path

raiz = str(Path(__file__).resolve().parent.parent)
if raiz not in sys.path:
    sys.path.insert(0, raiz)

from infraestrutura.configuracao_hf import configurar_huggingface
