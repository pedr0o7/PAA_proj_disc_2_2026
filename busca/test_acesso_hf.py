import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import HfHubHTTPError

import configuracao_hf as bootstrap
from infraestrutura import huggingface as acesso_hf


class TestAcessoHub(unittest.TestCase):
    def test_offline_nao_verifica_token_ou_rede(self):
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", True), \
             patch.object(acesso_hf, "get_token") as token, \
             patch.object(acesso_hf, "HfApi") as api:
            self.assertTrue(acesso_hf.preparar_acesso_hub())
            token.assert_not_called()
            api.assert_not_called()

    def test_sem_token_usa_acesso_publico(self):
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             patch.object(acesso_hf, "get_token", return_value=None), \
             patch.object(acesso_hf, "HfApi") as api:
            self.assertFalse(acesso_hf.preparar_acesso_hub())
            api.assert_not_called()

    def test_token_valido_e_verificado(self):
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             patch.object(acesso_hf, "get_token", return_value="token_de_teste"), \
             patch.object(acesso_hf, "HfApi") as api:
            with self.assertLogs("busca.hub", level="INFO") as logs:
                self.assertFalse(acesso_hf.preparar_acesso_hub())
            api.return_value.whoami.assert_called_once_with(cache=True)
            self.assertNotIn("token_de_teste", "\n".join(logs.output))

    def test_token_invalido_nao_recai_em_anonimo(self):
        resposta = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co"))
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             patch.object(acesso_hf, "get_token", return_value="token_de_teste"), \
             patch.object(acesso_hf, "HfApi") as api:
            api.return_value.whoami.side_effect = HfHubHTTPError("inválido", response=resposta)
            with self.assertRaisesRegex(RuntimeError, "Credencial .* inválida"):
                acesso_hf.preparar_acesso_hub()

    def test_falha_de_rede_forca_cache_local_e_avisa(self):
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             patch.object(acesso_hf, "get_token", return_value="token_de_teste"), \
             patch.object(acesso_hf, "HfApi") as api:
            api.return_value.whoami.side_effect = httpx.ConnectError("sem rede")
            with self.assertLogs("busca.hub", level="WARNING") as logs:
                self.assertTrue(acesso_hf.preparar_acesso_hub())
            self.assertIn("Credencial não verificada", "\n".join(logs.output))

    def test_erro_do_servidor_nao_ativa_offline(self):
        resposta = httpx.Response(503, request=httpx.Request("GET", "https://huggingface.co"))
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             patch.object(acesso_hf, "get_token", return_value="token_de_teste"), \
             patch.object(acesso_hf, "HfApi") as api:
            api.return_value.whoami.side_effect = HfHubHTTPError("indisponível", response=resposta)
            with self.assertRaises(HfHubHTTPError):
                acesso_hf.preparar_acesso_hub()

    def test_modelo_ausente_mostra_orientacao(self):
        with self.assertRaisesRegex(OSError, "cache do modelo está completo"):
            acesso_hf.carregar_modelo("o modelo", lambda: (_ for _ in ()).throw(OSError("ausente")))


if __name__ == "__main__":
    unittest.main()
