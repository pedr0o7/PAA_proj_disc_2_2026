import json
import tempfile
import unittest
from pathlib import Path

from corpus import amostrar_passagens


class AmostrarPassagensTest(unittest.TestCase):
    def test_amostra_no_maximo_uma_passagem_por_livro(self):
        with tempfile.TemporaryDirectory() as tmp:
            dados = Path(tmp)
            conteudo = ""
            livros = []
            for id_ in (1, 2):
                texto = (
                    f"Book {id_} first paragraph has enough words and a sentence ending. " * 8
                    + "\n\n"
                    + f"Book {id_} second paragraph also has enough words and a sentence ending. " * 8
                )
                inicio = len(conteudo.encode("utf-8"))
                conteudo += texto + "\n\n"
                livros.append({
                    "id": id_,
                    "titulo": f"Book {id_}",
                    "autores": "Author",
                    "url": f"https://www.gutenberg.org/ebooks/{id_}",
                    "deslocamento": inicio,
                    "bytes": len(texto.encode("utf-8")),
                })
            (dados / "textos.txt").write_bytes(conteudo.encode("utf-8"))
            (dados / "livros.jsonl").write_text(
                "".join(json.dumps(livro) + "\n" for livro in livros), encoding="utf-8"
            )

            passagens = amostrar_passagens(10, 4, dados, min_car=100, max_car=1000)

            self.assertEqual(len(passagens), 2)
            self.assertEqual(len({p["id"] for p in passagens}), len(passagens))
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from configuracao_hf import configurar_huggingface

configurar_huggingface()

import httpx
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from infraestrutura import huggingface as hub, wikipedia
from infraestrutura.integridade import validar_arquivo


class TestDownloadCorpus(unittest.TestCase):
    def setUp(self):
        diretorio = tempfile.TemporaryDirectory()
        self.addCleanup(diretorio.cleanup)
        self.legado = patch.object(wikipedia, "CACHE_MANIFESTO_ANTIGO", Path(diretorio.name) / "legado.json")
        self.legado.start()
        self.addCleanup(self.legado.stop)

    def criar_manifesto(self, conteudo):
        sha256 = hashlib.sha256(conteudo).hexdigest()
        return [
            SimpleNamespace(
                path=f"20231101.en/train-{i:05d}-of-00041.parquet",
                size=len(conteudo),
                lfs=SimpleNamespace(sha256=sha256),
            )
            for i in range(41)
        ]

    def test_reutiliza_cache_e_baixa_shard_ausente_pelo_hub(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            existente = pasta / "train-00000-of-00041.parquet"
            existente.write_bytes(conteudo)
            baixado = pasta / "download.parquet"
            baixado.write_bytes(conteudo)
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", pasta / "manifesto.json"), \
                 patch.object(hub, "HF_HUB_OFFLINE", False), \
                 patch.object(hub, "HfApi") as api, \
                 patch.object(hub, "hf_hub_download", return_value=str(baixado)) as download, \
                 patch.dict(os.environ, {"HF_TOKEN": "token_de_teste"}):
                api.return_value.list_repo_tree.return_value = self.criar_manifesto(conteudo)
                with self.assertLogs("busca.corpus", level="INFO") as logs:
                    self.assertEqual(wikipedia.baixar_shards(2), [existente, baixado])
                mensagens = "\n".join(logs.output)
                self.assertIn("Reutilizando shard local", mensagens)
                self.assertIn("Procurando shard", mensagens)
                self.assertEqual(mensagens.count("validado por tamanho e SHA-256"), 2)
                self.assertNotIn("token_de_teste", mensagens)
                download.assert_called_once_with(
                    repo_id=wikipedia.REPOSITORIO,
                    filename="20231101.en/train-00001-of-00041.parquet",
                    repo_type="dataset",
                    revision=wikipedia.REVISAO_HF,
                    local_files_only=False,
                )
                api.return_value.list_repo_tree.assert_called_once_with(
                    wikipedia.REPOSITORIO, path_in_repo=wikipedia.PASTA_SHARDS,
                    revision=wikipedia.REVISAO_HF, repo_type="dataset",
                )
                self.assertTrue((pasta / "manifesto.json").is_file())

    def test_offline_usa_manifesto_e_shard_validados_sem_rede(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            manifesto = pasta / "manifesto.json"
            itens = self.criar_manifesto(conteudo)
            manifesto.write_text(json.dumps([
                {"path": item.path, "size": item.size, "sha256": item.lfs.sha256}
                for item in itens
            ]), encoding="utf-8")
            shard = pasta / "train-00000-of-00041.parquet"
            shard.write_bytes(conteudo)
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", manifesto), \
                 patch.object(hub, "HfApi") as api, \
                 patch.object(hub, "hf_hub_download") as download:
                self.assertEqual(wikipedia.baixar_shards(1, offline=True), [shard])
                api.assert_not_called()
                download.assert_not_called()

    def test_offline_aceita_manifesto_no_local_antigo(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            manifesto_antigo = pasta / "legado.json"
            manifesto_antigo.write_text(json.dumps([
                {"path": item.path, "size": item.size, "sha256": item.lfs.sha256}
                for item in self.criar_manifesto(conteudo)
            ]), encoding="utf-8")
            shard = pasta / "train-00000-of-00041.parquet"
            shard.write_bytes(conteudo)
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", pasta / "novo.json"), \
                 patch.object(wikipedia, "CACHE_MANIFESTO_ANTIGO", manifesto_antigo):
                self.assertEqual(wikipedia.baixar_shards(1, offline=True), [shard])
                self.assertTrue(manifesto_antigo.exists())

    def test_falha_de_rede_usa_cache_do_hub_sem_nova_requisicao(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            manifesto = pasta / "manifesto.json"
            itens = self.criar_manifesto(conteudo)
            manifesto.write_text(json.dumps([
                {"path": item.path, "size": item.size, "sha256": item.lfs.sha256}
                for item in itens
            ]), encoding="utf-8")
            shard = pasta / "download.parquet"
            shard.write_bytes(conteudo)
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", manifesto), \
                 patch.object(hub, "HF_HUB_OFFLINE", False), \
                 patch.object(hub, "HfApi") as api, \
                 patch.object(hub, "hf_hub_download", return_value=str(shard)) as download:
                api.return_value.list_repo_tree.side_effect = httpx.ConnectError("sem rede")
                with self.assertLogs("busca.corpus", level="WARNING"):
                    self.assertEqual(wikipedia.baixar_shards(1), [shard])
                self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_offline_informa_manifesto_ou_shard_ausente(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            manifesto = pasta / "manifesto.json"
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", manifesto), \
                 patch.object(hub, "hf_hub_download", side_effect=LocalEntryNotFoundError("ausente")):
                with self.assertRaisesRegex(FileNotFoundError, "Manifesto local ausente"):
                    wikipedia.baixar_shards(1, offline=True)
                manifesto.write_text(json.dumps([
                    {"path": item.path, "size": item.size, "sha256": item.lfs.sha256}
                    for item in self.criar_manifesto(conteudo)
                ]), encoding="utf-8")
                with self.assertRaisesRegex(FileNotFoundError, "Shard .* não está completo"):
                    wikipedia.baixar_shards(1, offline=True)

    def test_falha_ao_baixar_shard_tenta_apenas_cache_local(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            shard = pasta / "download.parquet"
            shard.write_bytes(conteudo)
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", pasta / "manifesto.json"), \
                 patch.object(hub, "HF_HUB_OFFLINE", False), \
                 patch.object(hub, "HfApi") as api, \
                 patch.object(hub, "hf_hub_download", side_effect=[
                     httpx.ConnectError("sem rede"), str(shard)
                 ]) as download:
                api.return_value.list_repo_tree.return_value = self.criar_manifesto(conteudo)
                with self.assertLogs("busca.corpus", level="WARNING"):
                    self.assertEqual(wikipedia.baixar_shards(1), [shard])
                self.assertFalse(download.call_args_list[0].kwargs["local_files_only"])
                self.assertTrue(download.call_args_list[1].kwargs["local_files_only"])

    def test_erro_http_nao_usa_manifesto_local(self):
        resposta = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co"))
        with patch.object(hub, "HF_HUB_OFFLINE", False), \
             patch.object(hub, "HfApi") as api:
            api.return_value.list_repo_tree.side_effect = HfHubHTTPError("não autorizado", response=resposta)
            with self.assertRaises(HfHubHTTPError):
                wikipedia.baixar_shards(1)

    def test_rejeita_shard_com_checksum_incorreto(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "shard.parquet"
            caminho.write_bytes(b"shard")
            with self.assertRaisesRegex(ValueError, "Checksum incorreto"):
                validar_arquivo(caminho, "sha256", hashlib.sha256(b"outro").hexdigest(), 5)

    def test_baixar_shards_propaga_offline_entre_shards_e_no_retorno(self):
        conteudo = b"shard"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            shard0 = pasta / "s0.parquet"
            shard1 = pasta / "s1.parquet"
            shard0.write_bytes(conteudo)
            shard1.write_bytes(conteudo)
            with patch.object(wikipedia, "CACHE_PARQUET", pasta), \
                 patch.object(wikipedia, "CACHE_MANIFESTO", pasta / "manifesto.json"), \
                 patch.object(hub, "HF_HUB_OFFLINE", False), \
                 patch.object(hub, "HfApi") as api, \
                 patch.object(hub, "hf_hub_download", side_effect=[
                     httpx.ConnectError("sem rede"), str(shard0), str(shard1)
                 ]) as download:
                api.return_value.list_repo_tree.return_value = self.criar_manifesto(conteudo)
                with self.assertLogs("busca.corpus", level="WARNING"):
                    arquivos, offline = wikipedia.baixar_shards(2, retornar_offline=True)
            self.assertEqual(arquivos, [shard0, shard1])
            self.assertTrue(offline)
            self.assertFalse(download.call_args_list[0].kwargs["local_files_only"])
            self.assertTrue(download.call_args_list[1].kwargs["local_files_only"])
            self.assertTrue(download.call_args_list[2].kwargs["local_files_only"])



if __name__ == "__main__":
    unittest.main()
