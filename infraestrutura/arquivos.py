"""Gravação atômica de arquivos gerados."""
import os
import tempfile
from pathlib import Path


def gravar_atomico(caminho, escrever):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, nome = tempfile.mkstemp(prefix=".cache-", dir=caminho.parent)
    temporario = Path(nome)
    try:
        with os.fdopen(fd, "wb") as arquivo:
            escrever(arquivo)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        temporario.replace(caminho)
    finally:
        temporario.unlink(missing_ok=True)
