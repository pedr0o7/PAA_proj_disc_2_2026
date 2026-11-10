"""Compara métodos de busca semântica sobre passagens de dados/wikipedia_en_10gb/textos.txt.

Métodos: TF-IDF + cosseno (linha de base lexical), Word2Vec médio + cosseno,
Sentence Embeddings + cosseno (força bruta) e Sentence Embeddings + HNSW.
A avaliação usa consultas sintéticas: uma frase de uma passagem, sendo essa a
passagem relevante. Mede qualidade, tempo de construção e latência por consulta.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
TEXTOS = RAIZ / "dados" / "wikipedia_en_10gb" / "textos.txt"
TOKEN = re.compile(r"[a-z0-9]+")
FRASES = re.compile(r"(?<=[.!?])\s+")


def amostrar_passagens(caminho, n, semente, min_car=300, max_car=1200):
    """Sorteia passagens (blocos separados por linha em branco) em posições aleatórias do arquivo."""
    rng = random.Random(semente)
    tamanho = caminho.stat().st_size
    vistas, passagens = set(), []
    with open(caminho, "rb") as f:
        while len(passagens) < n:
            f.seek(rng.randrange(tamanho - 200_000))
            f.readline()
            bloco = f.read(200_000).decode("utf-8", errors="ignore")
            candidatas = [p.strip() for p in bloco.split("\n\n")[1:-1]]
            candidatas = [p for p in candidatas if min_car <= len(p) <= max_car]
            if not candidatas:
                continue
            for p in rng.sample(candidatas, min(3, len(candidatas))):
                if p not in vistas and len(passagens) < n:
                    vistas.add(p)
                    passagens.append(p)
    return passagens


def criar_consultas(passagens, q, semente):
    """Consulta = última frase com pelo menos 8 palavras; relevante = a passagem de origem."""
    rng = random.Random(semente + 1)
    indices = list(range(len(passagens)))
    rng.shuffle(indices)
    consultas = []
    for i in indices:
        frases = [s for s in FRASES.split(passagens[i]) if len(s.split()) >= 8]
        if len(frases) >= 2:
            consultas.append((frases[-1], i))
        if len(consultas) == q:
            break
    return consultas


def avaliar(topos, alvos, k=10):
    """topos: lista de listas de índices ordenados por relevância."""
    r1 = r10 = mrr = 0.0
    for lista, alvo in zip(topos, alvos):
        lista = [int(x) for x in lista[:k]]
        if alvo in lista:
            pos = lista.index(alvo)
            r10 += 1
            mrr += 1 / (pos + 1)
            r1 += pos == 0
    n = len(alvos)
    return {"recall@1": r1 / n, "recall@10": r10 / n, "mrr@10": mrr / n}


def cronometrar(funcao, entradas):
    """Latência por consulta, uma de cada vez (cenário interativo)."""
    tempos, saidas = [], []
    for e in entradas:
        t = time.perf_counter()
        saidas.append(funcao(e))
        tempos.append((time.perf_counter() - t) * 1000)
    return saidas, {"lat_media_ms": float(np.mean(tempos)), "lat_p95_ms": float(np.percentile(tempos, 95))}


def topk_cosseno(matriz, q, k):
    """Vetores já normalizados: cosseno = produto interno. Custo O(N*d) por consulta."""
    sims = matriz @ q
    idx = np.argpartition(-sims, k)[:k]
    return idx[np.argsort(-sims[idx])]


def normalizar(m):
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1
    return m / n


def metodo_tfidf(passagens, consultas, k):
    from sklearn.feature_extraction.text import TfidfVectorizer

    t = time.perf_counter()
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", dtype=np.float32)
    mat = vec.fit_transform(passagens)
    construcao = time.perf_counter() - t

    def buscar(c):
        sims = (mat @ vec.transform([c]).T).toarray().ravel()
        idx = np.argpartition(-sims, k)[:k]
        return idx[np.argsort(-sims[idx])]

    topos, lat = cronometrar(buscar, [c for c, _ in consultas])
    return topos, {"construcao_s": construcao, "indice_mb": (mat.data.nbytes + mat.indices.nbytes) / 2**20, **lat}


def metodo_word2vec(passagens, consultas, k, dim=100, epocas=5):
    from gensim.models import Word2Vec

    t = time.perf_counter()
    docs = [TOKEN.findall(p.lower()) for p in passagens]
    modelo = Word2Vec(docs, vector_size=dim, window=5, min_count=2, workers=4, epochs=epocas, seed=1)
    kv = modelo.wv
    # Média ponderada por IDF; sem ela as palavras comuns dominam a média.
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = len(docs)
    idf = {w: np.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def vetor(tokens):
        ws = [w for w in tokens if w in kv.key_to_index]
        if not ws:
            return np.zeros(dim, dtype=np.float32)
        pesos = np.array([idf.get(w, 1.0) for w in ws], dtype=np.float32)
        return (kv[ws] * pesos[:, None]).sum(0) / pesos.sum()

    mat = normalizar(np.stack([vetor(d) for d in docs]).astype(np.float32))
    construcao = time.perf_counter() - t

    def buscar(c):
        return topk_cosseno(mat, normalizar(vetor(TOKEN.findall(c.lower()))), k)

    topos, lat = cronometrar(buscar, [c for c, _ in consultas])
    return topos, {"construcao_s": construcao, "indice_mb": mat.nbytes / 2**20, **lat}


def metodo_sentence(passagens, consultas, k, modelo_nome, tamanho_lote=64):
    from sentence_transformers import SentenceTransformer
    from usearch.index import Index

    modelo = SentenceTransformer(modelo_nome, device="cpu")
    modelo.max_seq_length = 128
    t = time.perf_counter()
    mat = modelo.encode(passagens, batch_size=tamanho_lote, normalize_embeddings=True, show_progress_bar=False)
    mat = mat.astype(np.float32)
    codificacao = time.perf_counter() - t

    def consulta_vet(c):
        return modelo.encode([c], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    textos = [c for c, _ in consultas]
    # A codificação da consulta entra na latência total, pois faz parte do uso real.
    topos_bf, lat_bf = cronometrar(lambda c: topk_cosseno(mat, consulta_vet(c), k), textos)
    res_bf = {"construcao_s": codificacao, "indice_mb": mat.nbytes / 2**20, **lat_bf}

    t = time.perf_counter()
    indice = Index(ndim=mat.shape[1], metric="cos", connectivity=16, expansion_add=128, expansion_search=64)
    indice.add(np.arange(len(mat)), mat)
    constr_hnsw = time.perf_counter() - t
    topos_h, lat_h = cronometrar(lambda c: indice.search(consulta_vet(c), k).keys, textos)
    res_h = {"construcao_s": codificacao + constr_hnsw, "construcao_grafo_s": constr_hnsw,
             "indice_mb": mat.nbytes / 2**20 + indice.memory_usage / 2**20, **lat_h}

    # Latência só da busca vetorial (sem codificar a consulta), para isolar a complexidade.
    qs = list(np.stack([consulta_vet(c) for c in textos]))
    _, lat_bf_puro = cronometrar(lambda q: topk_cosseno(mat, q, k), qs)
    _, lat_h_puro = cronometrar(lambda q: indice.search(q, k).keys, qs)
    res_bf["busca_pura_ms"] = lat_bf_puro["lat_media_ms"]
    res_h["busca_pura_ms"] = lat_h_puro["lat_media_ms"]

    exato = [topk_cosseno(mat, q, k) for q in qs]
    aprox = [indice.search(q, k).keys for q in qs]
    res_h["recall_vs_exato@10"] = float(np.mean([len(set(a) & set(e)) / k for a, e in zip(aprox, exato)]))
    return topos_bf, res_bf, topos_h, res_h


def executar(n, q, k, semente, modelo_nome):
    passagens = amostrar_passagens(TEXTOS, n, semente)
    consultas = criar_consultas(passagens, q, semente)
    alvos = [i for _, i in consultas]
    resultados = {}

    def registrar(nome, topos, info):
        resultados[nome] = {**avaliar(topos, alvos, k), **info}
        print(nome, json.dumps(resultados[nome]), flush=True)

    registrar("tfidf_cosseno", *metodo_tfidf(passagens, consultas, k))
    registrar("word2vec_medio_cosseno", *metodo_word2vec(passagens, consultas, k))
    tb, rb, th, rh = metodo_sentence(passagens, consultas, k, modelo_nome)
    registrar("sentence_cosseno_bruto", tb, rb)
    registrar("sentence_hnsw", th, rh)
    return {"n_passagens": len(passagens), "n_consultas": len(consultas), "k": k, "resultados": resultados}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanhos", type=int, nargs="+", default=[2000, 8000, 20000])
    ap.add_argument("--consultas", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--modelo", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--saida", type=Path, default=Path(__file__).parent / "resultados.json")
    a = ap.parse_args()
    todos = []
    for n in a.tamanhos:
        print(f"== N={n}", flush=True)
        todos.append(executar(n, a.consultas, a.k, a.semente, a.modelo))
    a.saida.write_text(json.dumps(todos, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
