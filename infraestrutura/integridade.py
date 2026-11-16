"""Validação de arquivos baixados."""
import hashlib


def hash_arquivo(caminho, algoritmo):
    digest = hashlib.new(algoritmo)
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def validar_arquivo(caminho, algoritmo, esperado, tamanho=None):
    if tamanho is not None and caminho.stat().st_size != tamanho:
        raise ValueError(f"Tamanho incorreto do arquivo: {caminho}")
    if hash_arquivo(caminho, algoritmo) != esperado:
        raise ValueError(f"Checksum incorreto do arquivo: {caminho}")
