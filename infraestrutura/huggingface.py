"""Política de autenticação e acesso ao cache do Hugging Face."""
import logging

from infraestrutura.configuracao_hf import configurar_huggingface

configurar_huggingface()

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.errors import HfHubHTTPError

LOG = logging.getLogger("busca.hub")


def preparar_acesso_hub():
    if HF_HUB_OFFLINE:
        LOG.info("Modo offline: usando apenas arquivos locais; credencial não será verificada.")
        return True
    if not get_token():
        LOG.info("Nenhuma credencial do Hub disponível; acesso público anônimo.")
        return False
    try:
        HfApi().whoami(cache=True)
    except (httpx.NetworkError, httpx.TimeoutException):
        LOG.warning("Rede indisponível; usando apenas caches locais. Credencial não verificada.")
        return True
    except HfHubHTTPError as exc:
        if exc.response.status_code in (401, 403):
            raise RuntimeError(
                "Credencial do Hugging Face inválida ou sem autorização. Verifique HF_TOKEN e o login da CLI."
            ) from exc
        raise
    LOG.info("Credencial do Hugging Face validada.")
    return False


def carregar_modelo(descricao, funcao, *args, **kwargs):
    try:
        return funcao(*args, **kwargs)
    except HfHubHTTPError:
        raise
    except OSError as exc:
        raise OSError(
            f"Não foi possível carregar {descricao}. Verifique a rede, as permissões e se o cache do modelo está completo."
        ) from exc
