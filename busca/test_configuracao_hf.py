import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

import configuracao_hf as bootstrap
from infraestrutura import configuracao_hf


class TestConfiguracaoHuggingFace(unittest.TestCase):
    def setUp(self):
        configuracao_hf.configurar_huggingface.cache_clear()

    def tearDown(self):
        configuracao_hf.configurar_huggingface.cache_clear()

    def test_carrega_token_do_dotenv(self):
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            (raiz / ".env").write_text("HF_TOKEN=token_do_arquivo\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(configuracao_hf, "__file__", str(raiz / "busca" / "configuracao_hf.py")), \
                 patch.object(configuracao_hf, "import_module"):
                configuracao_hf.configurar_huggingface()
                self.assertEqual(os.environ["HF_TOKEN"], "token_do_arquivo")

    def test_dotenv_respeita_ambiente_e_ativa_aceleracao(self):
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            (raiz / ".env").write_text("HF_TOKEN=token_do_arquivo\n", encoding="utf-8")
            with patch.dict(os.environ, {"HF_TOKEN": "token_do_ambiente"}, clear=True), \
                 patch.object(configuracao_hf, "__file__", str(raiz / "busca" / "configuracao_hf.py")), \
                 patch.object(configuracao_hf, "import_module"):
                configuracao_hf.configurar_huggingface()
                self.assertEqual(os.environ["HF_TOKEN"], "token_do_ambiente")
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")

    def test_sem_token_preserva_login_cli_e_opcao_de_desempenho(self):
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            (raiz / ".env").write_text("HF_XET_HIGH_PERFORMANCE=0\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(configuracao_hf, "__file__", str(raiz / "busca" / "configuracao_hf.py")), \
                 patch.object(configuracao_hf, "import_module"):
                configuracao_hf.configurar_huggingface()
                self.assertNotIn("HF_TOKEN", os.environ)
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "0")

    def test_sem_hf_xet_ativa_fallback_com_aviso(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(configuracao_hf, "load_dotenv"), \
             patch.object(configuracao_hf, "import_module", side_effect=ImportError):
            with self.assertWarnsRegex(RuntimeWarning, "sem aceleração"):
                configuracao_hf.configurar_huggingface()
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")

    def test_xet_desabilitado_emite_aviso(self):
        with patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"}, clear=True), \
             patch.object(configuracao_hf, "load_dotenv"), \
             patch.object(configuracao_hf, "import_module") as importar:
            with self.assertWarnsRegex(RuntimeWarning, "HTTP sem aceleração"):
                configuracao_hf.configurar_huggingface()
            importar.assert_not_called()

    def test_token_vazio_no_dotenv_preserva_login_cli_e_anonimo(self):
        for conteudo in ("HF_TOKEN=\nHF_XET_HIGH_PERFORMANCE=0\n", "HF_TOKEN=   \n"):
            configuracao_hf.configurar_huggingface.cache_clear()
            with tempfile.TemporaryDirectory() as diretorio:
                raiz = Path(diretorio)
                (raiz / ".env").write_text(conteudo, encoding="utf-8")
                with patch.dict(os.environ, {}, clear=True), \
                     patch.object(configuracao_hf, "__file__", str(raiz / "infraestrutura" / "configuracao_hf.py")), \
                     patch.object(configuracao_hf, "import_module"):
                    configuracao_hf.configurar_huggingface()
                    self.assertNotIn("HF_TOKEN", os.environ)

    def test_env_example_nao_define_token_ficticio(self):
        exemplo = Path(__file__).resolve().parent.parent / ".env.example"
        valores = dotenv_values(exemplo)
        self.assertEqual(valores.get("HF_TOKEN"), "")


if __name__ == "__main__":
    unittest.main()
