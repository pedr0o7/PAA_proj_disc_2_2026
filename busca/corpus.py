"""Amostragem de passagens da Wikipédia EN com título e URL."""
import json
import logging
import random

from configuracao_hf import configurar_huggingface

configurar_huggingface()

import pyarrow.parquet as pq

from cache import gravar_atomico, identidade
from infraestrutura.wikipedia import REVISAO_HF, baixar_shards

LOG = logging.getLogger("busca.corpus")
MIN_CAR = 300
MAX_CAR = 1200
POR_ARTIGO = 3


def amostrar_passagens_titulo(n, semente, n_shards=1, min_car=MIN_CAR, max_car=MAX_CAR,
                              por_artigo=POR_ARTIGO, offline=None):
    """Sorteia artigos e, de cada um, até `por_artigo` parágrafos de tamanho adequado."""
    artigos = []
    for arq in baixar_shards(n_shards, offline):
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


def verificar_cache_passagens(cache_dir, n, semente, n_shards, regenerar_cache):
    parametros = {
        "versao": 1, "revisao": REVISAO_HF, "n": n, "semente": semente,
        "n_shards": n_shards, "min_car": MIN_CAR, "max_car": MAX_CAR, "por_artigo": POR_ARTIGO,
    }
    arq = cache_dir / f"passagens_v1_{identidade(parametros)}.json"
    legado = cache_dir / f"passagens_titulo_{n}_{semente}_{n_shards}.json"
    if not arq.exists() and legado.exists() and not regenerar_cache:
        raise RuntimeError(
            f"Cache antigo preservado em {legado}, mas faltam metadados de amostragem. "
            "Use --regenerar-cache para criar um novo cache sem modificar o antigo."
        )
    return parametros, arq, legado


def carregar_passagens(cache_dir, n, semente, n_shards=1, offline=None, regenerar_cache=False):
    """Carrega ou cria passagens identificadas pela origem e pelos parâmetros."""
    parametros, arq, legado = verificar_cache_passagens(cache_dir, n, semente, n_shards, regenerar_cache)
    if arq.exists():
        registro = json.loads(arq.read_text(encoding="utf-8"))
        if (not isinstance(registro, dict) or registro.get("parametros") != parametros
                or not isinstance(registro.get("passagens"), list)
                or registro.get("sha256") != identidade(registro["passagens"])):
            raise ValueError(f"Cache de passagens inválido ou incompatível: {arq}")
        validar_passagens(registro["passagens"])
        LOG.info("Usando passagens em cache; não será necessário obter shards para o corpus.")
        return registro["passagens"]
    if legado.exists():
        LOG.warning("Cache antigo preservado, mas não reutilizado: faltam metadados de amostragem.")
    LOG.info("Cache de passagens ausente; preparando corpus a partir de %d shard(s).", n_shards)
    passagens = amostrar_passagens_titulo(
        n, semente, n_shards, min_car=MIN_CAR, max_car=MAX_CAR, por_artigo=POR_ARTIGO, offline=offline,
    )
    validar_passagens(passagens)
    registro = {"parametros": parametros, "sha256": identidade(passagens), "passagens": passagens}
    dados = json.dumps(registro, ensure_ascii=False).encode("utf-8")
    gravar_atomico(arq, lambda arquivo: arquivo.write(dados))
    LOG.info("Corpus preparado: %d passagens.", len(passagens))
    return passagens


def validar_passagens(passagens):
    if not passagens or any(
        not isinstance(item, dict)
        or any(not isinstance(item.get(campo), str) or not item[campo] for campo in ("texto", "titulo", "url"))
        for item in passagens
    ):
        raise ValueError("As passagens estão vazias ou contêm texto, título ou URL inválidos.")
