"""Escalabilidade da busca vetorial pura (sem codificar texto): força bruta O(N*d) vs HNSW ~O(log N).

Usa vetores sintéticos normalizados em agrupamentos (aproximam a estrutura de embeddings reais),
dimensão 384 como all-MiniLM-L6-v2. Mede latência por consulta, construção do grafo e recall do HNSW.
"""
import json
import time
from pathlib import Path

import numpy as np
from usearch.index import Index

from benchmark import normalizar, topk_cosseno

DIM, K, CONSULTAS = 384, 10, 200


def gerar(n, rng, centros):
    base = centros[rng.integers(0, len(centros), n)] + 0.6 * rng.standard_normal((n, DIM), dtype=np.float32)
    return normalizar(base.astype(np.float32))


def main():
    rng = np.random.default_rng(0)
    centros = rng.standard_normal((200, DIM), dtype=np.float32)
    saida = []
    for n in [10_000, 100_000, 500_000, 1_000_000]:
        mat = gerar(n, rng, centros)
        qs = gerar(CONSULTAS, rng, centros)
        t = time.perf_counter()
        idx = Index(ndim=DIM, metric="cos", connectivity=16, expansion_add=128, expansion_search=64)
        idx.add(np.arange(n), mat)
        constr = time.perf_counter() - t
        t = time.perf_counter()
        exato = [topk_cosseno(mat, q, K) for q in qs]
        bf = (time.perf_counter() - t) / CONSULTAS * 1000
        t = time.perf_counter()
        aprox = [idx.search(q, K).keys for q in qs]
        hn = (time.perf_counter() - t) / CONSULTAS * 1000
        rec = float(np.mean([len(set(a) & set(e)) / K for a, e in zip(aprox, exato)]))
        linha = {"n": n, "bruto_ms": bf, "hnsw_ms": hn, "hnsw_construcao_s": constr,
                 "recall@10": rec, "vetores_mb": mat.nbytes / 2**20, "hnsw_mb": idx.memory_usage / 2**20}
        print(json.dumps(linha), flush=True)
        saida.append(linha)
    (Path(__file__).parent / "escala.json").write_text(json.dumps(saida, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
