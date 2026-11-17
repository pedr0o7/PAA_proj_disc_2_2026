import unittest
from unittest.mock import patch

import configuracao_hf as bootstrap

bootstrap.configurar_huggingface()

import httpx
from huggingface_hub.errors import HfHubHTTPError

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

    def test_carregar_modelo_faz_fallback_offline_em_erro_de_rede(self):
        chamadas = []

        def carregador(nome, local_files_only=False):
            chamadas.append(local_files_only)
            if not local_files_only:
                raise httpx.ConnectError("sem rede")
            return f"carregado:{nome}"

        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             self.assertLogs("busca.hub", level="WARNING") as logs:
            modelo, offline = acesso_hf.carregar_modelo(
                "o modelo", carregador, "m", local_files_only=False, retornar_offline=True,
            )
        self.assertEqual(modelo, "carregado:m")
        self.assertTrue(offline)
        self.assertEqual(chamadas, [False, True])
        self.assertIn("cache local", "\n".join(logs.output))

    def test_carregar_modelo_faz_fallback_em_timeout_e_retorna_direto_por_padrao(self):
        chamadas = []

        def carregador(local_files_only=False):
            chamadas.append(local_files_only)
            if not local_files_only:
                raise httpx.ReadTimeout("timeout")
            return "ok"

        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             self.assertLogs("busca.hub", level="WARNING"):
            modelo = acesso_hf.carregar_modelo("o modelo", carregador, local_files_only=False)
        self.assertEqual(modelo, "ok")
        self.assertEqual(chamadas, [False, True])

    def test_carregar_modelo_nao_retenta_quando_ja_offline_ou_erro_http(self):
        chamadas = []

        def carregador_offline(local_files_only=False):
            chamadas.append(local_files_only)
            raise httpx.ConnectError("sem rede")

        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False):
            with self.assertRaisesRegex(OSError, "cache do modelo está completo"):
                acesso_hf.carregar_modelo("o modelo", carregador_offline, local_files_only=True)
        self.assertEqual(chamadas, [True])

        resposta = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co"))
        with self.assertRaises(HfHubHTTPError):
            acesso_hf.carregar_modelo(
                "o modelo",
                lambda local_files_only=False: (_ for _ in ()).throw(HfHubHTTPError("401", response=resposta)),
                local_files_only=False,
            )

    def test_carregar_modelo_falha_se_cache_local_tambem_ausente_apos_fallback(self):
        def carregador(local_files_only=False):
            if not local_files_only:
                raise httpx.ConnectError("sem rede")
            raise OSError("ausente no cache")

        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             self.assertLogs("busca.hub", level="WARNING"):
            with self.assertRaisesRegex(OSError, "cache do modelo está completo"):
                acesso_hf.carregar_modelo("o modelo", carregador, local_files_only=False, retornar_offline=True)

    def test_fallback_em_remote_protocol_error_e_contexto_de_rede(self):
        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             patch.object(acesso_hf, "get_token", return_value="token_de_teste"), \
             patch.object(acesso_hf, "HfApi") as api:
            api.return_value.whoami.side_effect = httpx.RemoteProtocolError("conexao encerrada")
            with self.assertLogs("busca.hub", level="WARNING"):
                self.assertTrue(acesso_hf.preparar_acesso_hub())

        def carregador_contexto(local_files_only=False):
            if not local_files_only:
                try:
                    raise httpx.ConnectError("sem rede")
                except httpx.ConnectError:
                    raise OSError("falha ao baixar") from ValueError("detalhe")
            return "cache-ok"

        with patch.object(acesso_hf, "HF_HUB_OFFLINE", False), \
             self.assertLogs("busca.hub", level="WARNING"):
            modelo, offline = acesso_hf.carregar_modelo(
                "o modelo", carregador_contexto, local_files_only=False, retornar_offline=True,
            )
        self.assertEqual(modelo, "cache-ok")
        self.assertTrue(offline)


if __name__ == "__main__":
    unittest.main()
