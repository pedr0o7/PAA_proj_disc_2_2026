import gzip
import hashlib
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from coletar_ingles import baixar_continuavel, baixar_fontes, coletar_ingles, construir_indice, obter_shard_coleta
from extrair_10gb_en import extrair as extrair_texto


def gravar_dump(caminho, linhas):
    with gzip.open(caminho, "wb") as arquivo:
        arquivo.write(b"INSERT INTO tabela VALUES\n")
        arquivo.write(b"".join(linhas))


class Resposta(io.BytesIO):
    def __init__(self, dados, status, intervalo=""):
        super().__init__(dados)
        self.status = status
        self.headers = {"Content-Range": intervalo}


class TestColetaIngles(unittest.TestCase):
    def test_indice_filtra_namespace_e_ciclos(self):
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            gravar_dump(pasta / "enwiki-20261101-page.sql.gz", [
                b"(1,14,'Geography',0),\n",
                b"(2,14,'Climate',0),\n",
                b"(3,14,'Deserts',0),\n",
                b"(100,0,'Climate',0),\n",
                b"(101,0,'Desert',0),\n",
                b"(102,4,'Project',0);\n",
            ])
            gravar_dump(pasta / "enwiki-20261101-linktarget.sql.gz", [
                b"(10,14,'Geography'),\n",
                b"(11,14,'Climate'),\n",
                b"(12,14,'Deserts');\n",
            ])
            gravar_dump(pasta / "enwiki-20261101-categorylinks.sql.gz", [
                b"(2,'s','data','','subcat',1,10),\n",
                b"(3,'s','data','','subcat',1,11),\n",
                b"(1,'s','data','','subcat',1,12),\n",
                b"(100,'s','data','','page',1,10),\n",
                b"(100,'s','data','','page',1,11),\n",
                b"(101,'s','data','','page',1,12),\n",
                b"(102,'s','data','','page',1,11);\n",
            ])
            indice = construir_indice(pasta, 1)
            with closing(sqlite3.connect(indice)) as conexao:
                artigos = {linha[0] for linha in conexao.execute("SELECT id FROM selecionadas")}
                categorias = dict(conexao.execute("SELECT id, profundidade FROM alcancaveis"))
            self.assertEqual(artigos, {100})
            self.assertEqual(categorias, {10: 0, 11: 1})
            self.assertEqual(construir_indice(pasta, 1), indice)

    def test_download_continua_e_valida_hash(self):
        dados = b"primeira parte e segunda parte"
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "fonte.bin"
            parcial = caminho.with_name(caminho.name + ".part")
            parcial.write_bytes(dados[:8])
            with patch("coletar_ingles.urlopen", return_value=Resposta(dados[8:], 206, "bytes 8-29/30")):
                baixar_continuavel(
                    "https://example.com/fonte", caminho, "sha256",
                    hashlib.sha256(dados).hexdigest(), len(dados), "contato",
                )
            self.assertEqual(caminho.read_bytes(), dados)
            self.assertFalse(parcial.exists())

    def test_continuacao_rejeita_intervalo_incorreto(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "fonte.bin"
            caminho.with_name(caminho.name + ".part").write_bytes(b"abc")
            with patch("coletar_ingles.urlopen", return_value=Resposta(b"def", 206, "bytes 0-2/6")):
                with self.assertRaises(RuntimeError):
                    baixar_continuavel(
                        "https://example.com/fonte", caminho, "sha256", "0" * 64, 6, "contato"
                    )

    def test_shard_legado_parcial_e_retomado_e_validado(self):
        dados = b"primeira parte e segunda parte"
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            nome = "train-00000-of-00041.parquet"
            destino = pasta / nome
            destino.with_name(nome + ".part").write_bytes(dados[:8])
            item = {"path": "20231101.en/" + nome, "sha256": hashlib.sha256(dados).hexdigest(),
                    "size": len(dados)}
            with patch("coletar_ingles.urlopen", return_value=Resposta(dados[8:], 206, "bytes 8-29/30")):
                resultado, offline = obter_shard_coleta(nome, item, pasta, "contato", False)
            self.assertEqual(resultado, destino)
            self.assertFalse(offline)
            self.assertEqual(destino.read_bytes(), dados)
            self.assertFalse(destino.with_name(nome + ".part").exists())

    def test_dump_offline_nao_retomado_quando_incompleto(self):
        with tempfile.TemporaryDirectory() as diretorio:
            destino = Path(diretorio) / "fonte.bin"
            parcial = destino.with_name(destino.name + ".part")
            parcial.write_bytes(b"abc")
            with patch("coletar_ingles.urlopen") as abrir:
                with self.assertRaisesRegex(FileNotFoundError, "Arquivo completo ausente"):
                    baixar_continuavel("https://example.com", destino, "sha256", "0" * 64, 6, "contato", True)
            abrir.assert_not_called()
            self.assertEqual(parcial.read_bytes(), b"abc")

    def test_coleta_usa_caminhos_de_shards_no_cache_do_hub(self):
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            indice = pasta / "indice.sqlite"
            with closing(sqlite3.connect(indice)) as conexao:
                conexao.execute("CREATE TABLE selecionadas (id INTEGER)")
                conexao.execute("INSERT INTO selecionadas VALUES (1)")
                conexao.execute("CREATE TABLE alcancaveis (id INTEGER, profundidade INTEGER)")
                conexao.execute("INSERT INTO alcancaveis VALUES (2, 0)")
                conexao.commit()
            shard_hub = pasta / "hub" / "snapshot.parquet"
            with patch("coletar_ingles.construir_indice", return_value=indice), \
                 patch("coletar_ingles.extrair", return_value={}) as extrair:
                coletar_ingles(pasta, {"train-00000-of-00041.parquet": shard_hub}, pasta / "saida", 2)
            self.assertEqual(extrair.call_args.args[0], [shard_hub])

    def test_baixar_fontes_usa_acesso_e_manifesto_compartilhados(self):
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            item = {"path": "20231101.en/shard.parquet", "sha256": "0" * 64, "size": 1}
            shard = pasta / "hub" / "snapshot.parquet"
            with patch("coletar_ingles.DUMPS", {}), \
                 patch("coletar_ingles.preparar_acesso_hub", return_value=True) as preparar, \
                 patch("coletar_ingles.wikipedia.obter_manifesto", return_value=({"shard": item}, True)) as manifesto, \
                 patch("coletar_ingles.obter_shard_coleta", return_value=(shard, True)) as obter:
                self.assertEqual(baixar_fontes(pasta, "contato"), {"shard": shard})
            preparar.assert_called_once()
            manifesto.assert_called_once_with(True, contato="contato")
            obter.assert_called_once_with("shard", item, pasta, "contato", True)

    def test_extracao_usa_shard_compartilhado_e_modo_offline(self):
        with tempfile.TemporaryDirectory() as diretorio:
            pasta = Path(diretorio)
            shard = pasta / "snapshot.parquet"
            pq.write_table(pa.Table.from_pylist([{"id": "1", "text": "texto"}]), shard)
            item = {"path": "20231101.en/train-00000-of-00041.parquet", "sha256": "0" * 64, "size": 1}
            with patch("extrair_10gb_en.preparar_acesso_hub", return_value=True), \
                 patch("extrair_10gb_en.wikipedia.obter_manifesto", return_value=({"shard": item}, True)), \
                 patch("extrair_10gb_en.obter_shard_coleta", return_value=(shard, True)) as obter:
                extrair_texto("contato", cache=pasta / "cache", saida=pasta / "saida", meta=5)
            self.assertTrue(obter.call_args.args[-1])
            self.assertEqual((pasta / "saida" / "textos.txt").read_bytes(), b"texto\n\n")


if __name__ == "__main__":
    unittest.main()
