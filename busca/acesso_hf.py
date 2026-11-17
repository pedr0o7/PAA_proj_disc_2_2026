"""Compatibilidade para os scripts de busca."""
from configuracao_hf import configurar_huggingface

configurar_huggingface()

from infraestrutura.huggingface import carregar_modelo, preparar_acesso_hub
