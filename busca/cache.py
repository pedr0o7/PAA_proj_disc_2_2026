"""Identidade e gravação segura dos caches derivados."""
import hashlib
import json

from infraestrutura.arquivos import gravar_atomico


def identidade(valor):
    dados = json.dumps(valor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(dados).hexdigest()
