"""Revisão, manifesto e acesso aos shards da Wikipédia em inglês."""
import json
import logging
import re
from pathlib import Path

from infraestrutura.arquivos import gravar_atomico
from infraestrutura.integridade import validar_arquivo

REVISAO_HF = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
REPOSITORIO = "wikimedia/wikipedia"
PASTA_SHARDS = "20231101.en"
RAIZ = Path(__file__).resolve().parent.parent
CACHE_PARQUET = RAIZ / "dados" / "cache_ingles"
CACHE_MANIFESTO = CACHE_PARQUET / f"manifesto_wikipedia_{REVISAO_HF}.json"
CACHE_MANIFESTO_ANTIGO = RAIZ / "busca" / "cache_indice" / f"manifesto_wikipedia_{REVISAO_HF}.json"
LOG = logging.getLogger("busca.corpus")


def validar_manifesto(itens):
    if not isinstance(itens, list) or len(itens) != 41:
        raise ValueError("O manifesto deve conter 41 arquivos ingleses.")
    disponiveis = {}
    for item in itens:
        if not isinstance(item, dict):
            raise ValueError("Entrada inválida no manifesto do dataset.")
        caminho = item.get("path")
        tamanho = item.get("size")
        sha256 = item.get("sha256")
        if (not isinstance(caminho, str)
                or not re.fullmatch(r"20231101\.en/train-\d{5}-of-00041\.parquet", caminho)
                or type(tamanho) is not int or tamanho <= 0
                or not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256)):
            raise ValueError(f"Entrada inválida no manifesto do dataset: {caminho!r}")
        nome = caminho.rsplit("/", 1)[-1]
        disponiveis[nome] = item
    if len(disponiveis) != 41:
        raise ValueError("O manifesto contém nomes duplicados.")
    if set(disponiveis) != {f"train-{i:05d}-of-00041.parquet" for i in range(41)}:
        raise ValueError("O manifesto não contém todos os shards esperados.")
    return disponiveis


def obter_manifesto(offline=None, contato=None):
    from infraestrutura import huggingface as hub

    offline = hub.HF_HUB_OFFLINE if offline is None else offline or hub.HF_HUB_OFFLINE
    if not offline:
        LOG.info("Consultando o manifesto do corpus no Hugging Face.")
        try:
            api = hub.HfApi(user_agent=f"ProjetoGeografia/1.0 ({contato})") if contato else hub.HfApi()
            itens_hub = list(api.list_repo_tree(
                REPOSITORIO, path_in_repo=PASTA_SHARDS, revision=REVISAO_HF, repo_type="dataset",
            ))
        except hub.ERROS_DE_REDE:
            LOG.warning("Rede indisponível ao consultar o manifesto; usando apenas caches locais.")
            offline = True
        else:
            itens = [
                {"path": item.path, "size": getattr(item, "size", None),
                 "sha256": getattr(getattr(item, "lfs", None), "sha256", None)}
                for item in itens_hub
            ]
            disponiveis = validar_manifesto(itens)
            CACHE_MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
            dados = json.dumps(itens).encode("utf-8")
            gravar_atomico(CACHE_MANIFESTO, lambda arquivo: arquivo.write(dados))
            LOG.info("Manifesto validado: %d shards disponíveis.", len(disponiveis))
            return disponiveis, False
    manifesto = CACHE_MANIFESTO if CACHE_MANIFESTO.is_file() else CACHE_MANIFESTO_ANTIGO
    if not manifesto.is_file():
        raise FileNotFoundError(
            f"Manifesto local ausente: {CACHE_MANIFESTO}. Prepare o corpus com rede antes de usar os shards offline."
        )
    disponiveis = validar_manifesto(json.loads(manifesto.read_text(encoding="utf-8")))
    LOG.info("Usando manifesto local do corpus.")
    LOG.info("Manifesto validado: %d shards disponíveis.", len(disponiveis))
    return disponiveis, True


def obter_shard(nome, item, cache, offline):
    from infraestrutura import huggingface as hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    offline = bool(offline) or hub.HF_HUB_OFFLINE
    destino = Path(cache) / nome
    if not destino.exists():
        LOG.info("Procurando shard %s no cache do Hub%s.", nome, "" if offline else " ou na rede")
        try:
            try:
                destino = Path(hub.hf_hub_download(
                    repo_id=REPOSITORIO, filename=item["path"], repo_type="dataset",
                    revision=REVISAO_HF, local_files_only=offline,
                ))
            except hub.ERROS_DE_REDE:
                if offline:
                    raise
                LOG.warning("Rede indisponível ao obter o shard %s; usando apenas cache local.", nome)
                offline = True
                destino = Path(hub.hf_hub_download(
                    repo_id=REPOSITORIO, filename=item["path"], repo_type="dataset",
                    revision=REVISAO_HF, local_files_only=True,
                ))
        except LocalEntryNotFoundError as exc:
            raise FileNotFoundError(
                f"Shard {nome} não está completo no cache local; conecte-se e prepare o corpus."
            ) from exc
    else:
        LOG.info("Reutilizando shard local %s.", nome)
    validar_arquivo(destino, "sha256", item["sha256"], item["size"])
    LOG.info("Shard %s validado por tamanho e SHA-256.", nome)
    return destino, offline


def baixar_shards(n_shards, offline=None, cache=None, retornar_offline=False):
    if not 1 <= n_shards <= 41:
        raise ValueError("O número de shards deve estar entre 1 e 41.")
    disponiveis, offline = obter_manifesto(offline)
    cache = CACHE_PARQUET if cache is None else cache
    arquivos = []
    for nome, item in sorted(disponiveis.items())[:n_shards]:
        destino, offline = obter_shard(nome, item, cache, offline)
        arquivos.append(destino)
    return (arquivos, offline) if retornar_offline else arquivos
