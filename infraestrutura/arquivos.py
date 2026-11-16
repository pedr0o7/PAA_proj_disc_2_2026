"""Gravação atômica de arquivos gerados."""
import os
import tempfile
from pathlib import Path


def gravar_atomico(caminho, escrever):
    fd, nome = tempfile.mkstemp(prefix=".cache-", dir=caminho.parent)
    temporario = Path(nome)
    try:
        with os.fdopen(fd, "wb") as arquivo:
            escrever(arquivo)
        temporario.replace(caminho)
    finally:
        temporario.unlink(missing_ok=True)
