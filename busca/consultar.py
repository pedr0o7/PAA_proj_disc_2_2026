"""Busca semântica interativa (sentence embeddings + HNSW) sobre passagens de textos.txt.

Na primeira execução amostra N passagens, gera os embeddings e salva em busca/cache_indice;
nas seguintes, carrega do cache. Uso: consultar.py [--n 5000] [-p "pergunta"]
"""
import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from usearch.index import Index

from benchmark import TEXTOS, amostrar_passagens

CACHE = Path(__file__).parent / "cache_indice"
MODELO = "sentence-transformers/all-MiniLM-L6-v2"


def carregar(n, semente, modelo):
    CACHE.mkdir(exist_ok=True)
    pass_arq = CACHE / f"passagens_{n}_{semente}.json"
    emb_arq = CACHE / f"emb_{n}_{semente}.npy"
    if pass_arq.exists() and emb_arq.exists():
        passagens = json.loads(pass_arq.read_text(encoding="utf-8"))
        return passagens, np.load(emb_arq)
    print(f"Primeira execução: codificando {n} passagens (~{n // 43 // 60 + 1} min em CPU)...")
    passagens = amostrar_passagens(TEXTOS, n, semente)
    modelo.max_seq_length = 128
    emb = modelo.encode(passagens, batch_size=64, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    pass_arq.write_text(json.dumps(passagens), encoding="utf-8")
    np.save(emb_arq, emb)
    return passagens, emb


def responder(pergunta, modelo, indice, passagens, k):
    q = modelo.encode([pergunta], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
    achados = indice.search(q, k)
    for pos, (chave, dist) in enumerate(zip(achados.keys, achados.distances), 1):
        print(f"\n[{pos}] similaridade {1 - dist:.3f}")
        print(textwrap.fill(passagens[int(chave)], 100, initial_indent="    ", subsequent_indent="    "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="passagens indexadas")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--k", type=int, default=3, help="resultados por pergunta")
    ap.add_argument("-p", "--pergunta", help="pergunta única (sem modo interativo)")
    a = ap.parse_args()

    modelo = SentenceTransformer(MODELO, device="cpu")
    passagens, emb = carregar(a.n, a.semente, modelo)
    indice = Index(ndim=emb.shape[1], metric="cos", connectivity=16, expansion_add=128, expansion_search=64)
    indice.add(np.arange(len(emb)), emb)

    if a.pergunta:
        responder(a.pergunta, modelo, indice, passagens, a.k)
        return
    print(f"{len(passagens)} passagens indexadas. Pergunte em inglês; linha vazia ou 'sair' encerra.")
    while True:
        pergunta = input("\nPergunta> ").strip()
        if pergunta.lower() in ("", "sair", "exit"):
            break
        responder(pergunta, modelo, indice, passagens, a.k)


if __name__ == "__main__":
    main()
