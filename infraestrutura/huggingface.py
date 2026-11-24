"""Política de autenticação e acesso ao cache do Hugging Face."""
import inspect
import logging

from infraestrutura.configuracao_hf import configurar_huggingface

configurar_huggingface()

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.errors import HfHubHTTPError

LOG = logging.getLogger("busca.hub")
ERROS_DE_REDE = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)


def preparar_acesso_hub():
    if HF_HUB_OFFLINE:
        LOG.info("Modo offline: usando apenas arquivos locais; credencial não será verificada.")
        return True
    if not get_token():
        LOG.info("Nenhuma credencial do Hub disponível; acesso público anônimo.")
        return False
    try:
        HfApi().whoami(cache=True)
    except ERROS_DE_REDE:
        LOG.warning("Rede indisponível; usando apenas caches locais. Credencial não verificada.")
        return True
    except HfHubHTTPError as exc:
        if getattr(exc.response, "status_code", None) in (401, 403):
            raise RuntimeError(
                "Credencial do Hugging Face inválida ou sem autorização. Verifique HF_TOKEN e o login da CLI."
            ) from exc
        raise
    LOG.info("Credencial do Hugging Face validada.")
    return False


def _aceita_local_files_only(funcao):
    try:
        parametros = inspect.signature(funcao).parameters
    except (TypeError, ValueError):
        return False
    return "local_files_only" in parametros or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parametros.values()
    )


def _erro_de_rede(exc):
    vistos = set()
    pendentes = [exc]
    while pendentes:
        atual = pendentes.pop()
        if atual is None or id(atual) in vistos:
            continue
        if isinstance(atual, ERROS_DE_REDE):
            return True
        vistos.add(id(atual))
        pendentes.extend((atual.__cause__, atual.__context__))
    return False


def _erro_carregar_modelo(descricao):
    return OSError(
        f"Não foi possível carregar {descricao}. Verifique a rede, as permissões e se o cache do modelo está completo."
    )


def carregar_modelo(descricao, funcao, *args, retornar_offline=False, **kwargs):
    offline = bool(kwargs.get("local_files_only", False)) or HF_HUB_OFFLINE
    if offline and "local_files_only" in kwargs:
        kwargs = {**kwargs, "local_files_only": True}
    try:
        modelo = funcao(*args, **kwargs)
    except HfHubHTTPError:
        raise
    except (*ERROS_DE_REDE, OSError) as exc:
        pode_tentar_local = (
            not offline
            and _erro_de_rede(exc)
            and ("local_files_only" in kwargs or _aceita_local_files_only(funcao))
        )
        if not pode_tentar_local:
            raise _erro_carregar_modelo(descricao) from exc
        LOG.warning("Rede indisponível ao carregar %s; usando apenas cache local.", descricao)
        offline = True
        try:
            modelo = funcao(*args, **{**kwargs, "local_files_only": True})
        except HfHubHTTPError:
            raise
        except (*ERROS_DE_REDE, OSError) as fallback_exc:
            raise _erro_carregar_modelo(descricao) from fallback_exc
    return (modelo, offline) if retornar_offline else modelo
