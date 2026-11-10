"""Passagens da Wikipédia EN com título/URL de origem (necessários para citar as fontes).

O textos.txt não guarda o título dos artigos; por isso baixamos o(s) primeiro(s) parquet(s) do mesmo
dataset (wikimedia/wikipedia, mesma revisão) e amostramos passagens mantendo {texto, titulo, url}.
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "dados"))
from coletar_ingles import BASE_HF, baixar_continuavel, manifesto  # noqa: E402

CACHE_PARQUET = RAIZ / "dados" / "cache_ingles"
CONTATO = "projeto-academico-paa"


def baixar_shards(n_shards):
    disponiveis = manifesto(CONTATO)
    CACHE_PARQUET.mkdir(exist_ok=True)
    arquivos = []
    for nome in sorted(disponiveis)[:n_shards]:
        caminho, sha256, tamanho = disponiveis[nome]
        destino = CACHE_PARQUET / nome
        baixar_continuavel(BASE_HF + caminho, destino, "sha256", sha256, tamanho, CONTATO)
        arquivos.append(destino)
    return arquivos


def amostrar_passagens_titulo(n, semente, n_shards=1, min_car=300, max_car=1200, por_artigo=3):
    """Sorteia artigos e, de cada um, até `por_artigo` parágrafos de tamanho adequado."""
    artigos = []
    for arq in baixar_shards(n_shards):
        for lote in pq.ParquetFile(arq).iter_batches(batch_size=2000, columns=["title", "url", "text"]):
            for titulo, url, texto in zip(lote.column("title").to_pylist(), lote.column("url").to_pylist(),
                                          lote.column("text").to_pylist()):
                paragrafos = [p.strip() for p in (texto or "").split("\n\n")]
                paragrafos = [p for p in paragrafos if min_car <= len(p) <= max_car]
                if paragrafos:
                    artigos.append((titulo, url, paragrafos))
    rng = random.Random(semente)
    rng.shuffle(artigos)
    passagens, vistas = [], set()
    for titulo, url, paragrafos in artigos:
        for p in rng.sample(paragrafos, min(por_artigo, len(paragrafos))):
            if p not in vistas:
                vistas.add(p)
                passagens.append({"texto": p, "titulo": titulo, "url": url})
        if len(passagens) >= n:
            break
    return passagens[:n]


def carregar_passagens(cache_dir, n, semente, n_shards=1):
    """Retorna lista de dicts {texto, titulo, url}; usa cache em disco se existir."""
    arq = cache_dir / f"passagens_titulo_{n}_{semente}_{n_shards}.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    passagens = amostrar_passagens_titulo(n, semente, n_shards)
    arq.write_text(json.dumps(passagens), encoding="utf-8")
    return passagens
